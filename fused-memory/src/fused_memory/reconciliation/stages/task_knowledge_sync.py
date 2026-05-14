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
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol

from fused_memory.middleware.task_interceptor import TERMINAL_STATUSES
from fused_memory.models.reconciliation import (
    ReconciliationEvent,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.cli_stage_runner import (
    DISALLOW_TASK_WRITES,
    STAGE2_DISALLOWED,
    STAGE3_DISALLOWED,
    STAGE3_REPORT_SCHEMA,
)
from fused_memory.reconciliation.flag_dedup import compute_flag_signature
from fused_memory.reconciliation.policies.autopilot_video import (
    AUTOPILOT_VIDEO_PROJECT_ID,
    AUTOPILOT_VIDEO_TASK_CEILING,
    excessive_autopilot_video_ids,
)
from fused_memory.reconciliation.prompts import (
    _STAGE2_PROJECT_ID_GUIDELINE,
    _STAGE3_PROJECT_ID_GUIDELINE,
)
from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.task_filter import (
    MAX_ACTIVE_TASKS_RENDERED,
    FilteredTaskTree,
    filter_task_tree,
    format_filtered_task_tree,
    format_task_list,
    id_key,
)

logger = logging.getLogger(__name__)


def _extract_status(task_data: dict) -> str:
    """Extract status from a Taskmaster get_task response dict."""
    if 'status' in task_data:
        return task_data['status']
    data = task_data.get('data', {})
    if isinstance(data, dict):
        return data.get('status', 'unknown')
    return 'unknown'


# Projects allowed to use the briefing-refresh hook.  This is a reify-specific
# feature; gating on project_id prevents accidental triggering by other projects
# that happen to have the same file layout.  Extend when needed.
_BRIEFING_REFRESH_PROJECT_ALLOWLIST: frozenset[str] = frozenset({'reify'})

# ── Task-1139 scope filter ────────────────────────────────────────────────────
# These substrings uniquely identify Mem0 flags that were written as part of the
# FIX-A bug-mechanics description itself.  They must not be re-surfaced as live
# flags for other tasks.  The list is intentionally narrow so legitimate flags
# that merely mention "Stage 1" or "flagged_items" in passing are not suppressed.
#
# TODO(task-1139-gc): Remove this constant and _should_skip_known_bug_1139_flag
# once the Mem0 collection no longer contains task_id=1139 flag_for_stage2
# memories.  Trigger: verify via count_memories_by_metadata(filters={'task_id':
# '1139', 'flag_for_stage2': True}) == 0, then delete both symbols.
_KNOWN_BUG_1139_CONTENT_MARKERS: tuple[str, ...] = (
    'flag_for_stage2=true but does NOT include them in flagged_items',
    'Stage 1 LLM writes flags to Mem0 with metadata.flag_for_stage2',
)

# Persistence-threshold constant for FIX D (stale-flag escalation guard).
# A flag that has survived this many Stage 2 cycles without being deleted is
# considered stale and is surfaced in the payload for operator escalation.
STAGE2_FLAG_PERSISTENCE_THRESHOLD: int = 3


def _should_skip_known_bug_1139_flag(flag: dict) -> bool:
    """Return True iff *flag* describes the task-1139 flag-relay bug mechanics.

    The active-Mem0-query path (FIX A) must not re-inject into the payload any
    flags that were written specifically to describe *this* bug.  Two narrow
    signals identify such a flag:

    1. ``flag['task_id'] == '1139'`` (string-coerced, so int 1139 also matches).
    2. ``flag['content']`` contains one of the ``_KNOWN_BUG_1139_CONTENT_MARKERS``
       substrings — e.g. the sentence "Stage 1 LLM writes flags to Mem0 with
       metadata.flag_for_stage2 but does NOT include them in flagged_items".

    All other flags pass through unchanged.

    .. note::
        This filter becomes dead code once task 1139 closes and its Mem0 flags
        are GC'd.  See the TODO on ``_KNOWN_BUG_1139_CONTENT_MARKERS`` for the
        removal trigger.
    """
    if str(flag.get('task_id', '')) == '1139':
        return True
    content = flag.get('content', '')
    return any(marker in content for marker in _KNOWN_BUG_1139_CONTENT_MARKERS)


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


async def _resolve_live_status(
    op: dict,
    taskmaster,
    project_root: str,
    status_cache: dict[str, str] | None,
    op_name: str,
    *,
    _parsed_params: dict | None = None,
) -> tuple[str, str] | None:
    """Return (task_id, live_status) for a write_journal op, or None to skip it.

    Centralizes the param-parse + task_id-extract + cache-or-fetch boilerplate
    shared by Guards 1-3. Returns ``None`` when:
      * ``op['params']`` is malformed JSON
      * task_id is missing (or empty after str-strip)
      * status_cache is provided but task_id is absent (cache-build failure)
      * fallback ``taskmaster.get_task`` raises

    The task_id source is keyed off ``op.get('operation')``:
      * ``'add_memory'``      → ``params['metadata']['task_id']``
      * anything else         → ``params['task_id']``

    In fallback mode (status_cache is None), a non-dict ``get_task`` result
    yields ``live_status='unknown'`` (preserves pre-refactor behaviour).

    Args:
        op: A single write_journal op dict. Caller is expected to have already
            filtered by ``agent_id`` and ``operation``.
        taskmaster: Taskmaster backend (used only in fallback mode).
        project_root: Absolute path to the Taskmaster project directory.
        status_cache: Pre-fetched ``{task_id: live_status}`` dict built by
            ``_apply_post_flight_guards``, or ``None`` for per-op fallback.
        op_name: Calling helper's name, embedded in WARNING log messages so
            existing log greps remain stable (e.g.
            'reconciliation._verify_set_task_status_post_action: ...').
        _parsed_params: Optional pre-parsed params dict.  When provided, the
            JSON-parse step is skipped entirely, avoiding a second
            ``json.loads`` call in callers that already parsed params locally
            (Guards 2 and 3).
    """
    if _parsed_params is not None:
        params = _parsed_params
    else:
        params_raw = op.get('params') or '{}'
        try:
            params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                'reconciliation.%s: failed to parse params JSON for op_id=%s; skipping',
                op_name,
                op.get('id'),
            )
            return None

    if op.get('operation') == 'add_memory':
        metadata = params.get('metadata') or {}
        if not isinstance(metadata, dict):
            return None
        task_id = str(metadata.get('task_id', '')).strip()
    else:
        task_id = str(params.get('task_id', '')).strip()
    if not task_id:
        return None

    if status_cache is not None:
        if task_id not in status_cache:
            return None  # cache build failed for this task; skip
        live_status = status_cache[task_id]
    else:
        try:
            task_data = await taskmaster.get_task(task_id, project_root)
        except Exception:
            logger.warning(
                'reconciliation.%s: get_task failed for task_id=%s; skipping op',
                op_name,
                task_id,
            )
            return None
        live_status = _extract_status(task_data) if isinstance(task_data, dict) else 'unknown'

    return task_id, live_status


async def _classify_terminal_state_violations(
    ops: list[dict],
    taskmaster,
    project_root: str,
    agent_id: str,
    status_cache: dict[str, str] | None = None,
) -> list[dict]:
    """Classify write_journal ops that mutated tasks already in a terminal state.

    For every ``update_task`` op authored by the Stage 2 agent, fetches the
    *live* task status from Taskmaster and returns a violation record when the
    status is in ``TERMINAL_STATUSES`` (done, cancelled).

    The "skip the write entirely" semantics described in the task specification
    cannot be enforced post-hoc (the LLM's write already landed), so this
    helper *reclassifies* the op instead.  Callers are expected to:

    - increment ``stats['not_applicable_count']`` by ``len(violations)``
    - decrement ``stats['tasks_modified']`` by ``len(violations)`` (clamped at 0)

    ops from ``agent_id='task-interceptor'`` are intentionally excluded; the
    interceptor performs legitimate ``update_task(metadata=...)`` calls during
    set_task_status (e.g. ``done_provenance`` audit-field merge).

    Args:
        ops: Write-journal op dicts returned by
            ``WriteJournal.get_ops_by_causation(run_id)`` (layer=='write_op').
        taskmaster: Taskmaster backend (must have an async ``get_task`` method).
        project_root: Absolute path to the Taskmaster project directory.
        agent_id: The agent_id string to filter ops on (derived from stage_id).
        status_cache: Pre-fetched ``{task_id: live_status}`` dict built by
            ``_apply_post_flight_guards``.  When provided, skips individual
            ``taskmaster.get_task`` calls.

    Returns:
        List of ``{'op_id', 'task_id', 'live_status', 'reason': 'not_applicable'}``
        dicts, one per qualifying violation.  Empty list when no violations are
        found.
    """
    violations: list[dict] = []
    for op in ops:
        if op.get('agent_id') != agent_id:
            continue
        if op.get('operation') != 'update_task':
            continue

        resolved = await _resolve_live_status(
            op,
            taskmaster,
            project_root,
            status_cache,
            '_classify_terminal_state_violations',
        )
        if resolved is None:
            continue
        task_id, live_status = resolved

        if live_status in TERMINAL_STATUSES:
            violations.append(
                {
                    'op_id': op.get('id'),
                    'task_id': task_id,
                    'live_status': live_status,
                    'reason': 'not_applicable',
                }
            )

    return violations


async def _verify_set_task_status_post_action(
    ops: list[dict],
    taskmaster,
    project_root: str,
    agent_id: str,
    status_cache: dict[str, str] | None = None,
) -> list[dict]:
    """Verify that each Stage 2 set_task_status op actually took effect.

    For every ``set_task_status`` op authored by the Stage 2 agent, fetches the
    *live* task status from Taskmaster and returns a mismatch record when the live
    status differs from the op's target ``status`` param.

    This catches cases where the task_interceptor's terminal-exit gate (or other
    guards) silently rejected the status transition but the LLM still inflated its
    self-reported ``tasks_modified`` count.

    Args:
        ops: Write-journal op dicts (``layer=='write_op'``).
        taskmaster: Taskmaster backend.
        project_root: Absolute path to the Taskmaster project directory.
        agent_id: The agent_id string to filter ops on (derived from stage_id).
        status_cache: Pre-fetched ``{task_id: live_status}`` dict built by
            ``_apply_post_flight_guards``.  When provided, skips individual
            ``taskmaster.get_task`` calls.

    Returns:
        List of ``{'op_id', 'task_id', 'target_status', 'live_status'}`` dicts.
    """
    mismatches: list[dict] = []
    for op in ops:
        if op.get('agent_id') != agent_id:
            continue
        if op.get('operation') != 'set_task_status':
            continue

        # Parse params locally to extract target_status; pass pre-parsed dict to the
        # helper to avoid a second json.loads call on the same small payload.
        params_raw = op.get('params') or '{}'
        try:
            params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                'reconciliation._verify_set_task_status_post_action: '
                'failed to parse params JSON for op_id=%s; skipping',
                op.get('id'),
            )
            continue
        target_status = str(params.get('status', '')).strip()
        if not target_status:
            continue

        resolved = await _resolve_live_status(
            op,
            taskmaster,
            project_root,
            status_cache,
            '_verify_set_task_status_post_action',
            _parsed_params=params,
        )
        if resolved is None:
            continue
        task_id, live_status = resolved

        if live_status != target_status:
            mismatches.append(
                {
                    'op_id': op.get('id'),
                    'task_id': task_id,
                    'target_status': target_status,
                    'live_status': live_status,
                }
            )

    return mismatches


# Metadata keys that mark an add_memory op for freshness checking.
# The LLM may emit either ``snapshot_status`` (canonical) or ``observed_status``
# (natural-language alias).  Both keys carry the same semantics: the task status
# that was observed *before* the LLM started writing this memory.  The guard
# fetches the *live* status and flags a mismatch if they differ.
_STAGE2_STALL_SNAPSHOT_KEYS: tuple[str, ...] = ('snapshot_status', 'observed_status')


async def _check_stall_guard_freshness(
    ops: list[dict],
    taskmaster,
    project_root: str,
    agent_id: str,
    status_cache: dict[str, str] | None = None,
) -> list[dict]:
    """Check that add_memory ops written against a snapshot status are still fresh.

    For every ``add_memory`` op authored by the Stage 2 agent whose
    ``params.metadata`` contains a ``snapshot_status`` (or ``observed_status``)
    key AND a ``task_id``, this helper fetches the *live* task status and flags
    a mismatch.

    A mismatch indicates that the memory was written against a stale snapshot —
    the LLM observed the task in one state, wrote a memory anchored to that
    state, but the task has since transitioned.

    Args:
        ops: Write-journal op dicts (``layer=='write_op'``).
        taskmaster: Taskmaster backend.
        project_root: Absolute path to the Taskmaster project directory.
        agent_id: The agent_id string to filter ops on (derived from stage_id).
        status_cache: Pre-fetched ``{task_id: live_status}`` dict built by
            ``_apply_post_flight_guards``.  When provided, skips individual
            ``taskmaster.get_task`` calls.

    Returns:
        List of ``{'op_id', 'task_id', 'snapshot_status', 'live_status'}`` dicts.
    """
    violations: list[dict] = []
    for op in ops:
        if op.get('agent_id') != agent_id:
            continue
        if op.get('operation') != 'add_memory':
            continue

        # Parse params locally to check for snapshot_status / observed_status keys
        # before calling _resolve_live_status — this preserves the early-continue that
        # prevents get_task from being called when the op has no freshness key.
        # The pre-parsed dict is then passed to the helper to avoid a second json.loads.
        params_raw = op.get('params') or '{}'
        try:
            params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                'reconciliation._check_stall_guard_freshness: '
                'failed to parse params JSON for op_id=%s; skipping',
                op.get('id'),
            )
            continue
        metadata = params.get('metadata') or {}
        if not isinstance(metadata, dict):
            continue

        # Resolve snapshot_status or its alias
        snapshot_status: str | None = None
        for key in _STAGE2_STALL_SNAPSHOT_KEYS:
            if key in metadata:
                snapshot_status = str(metadata[key])
                break
        if snapshot_status is None:
            continue  # Op not opted into freshness checking; skip before any get_task call

        resolved = await _resolve_live_status(
            op,
            taskmaster,
            project_root,
            status_cache,
            '_check_stall_guard_freshness',
            _parsed_params=params,
        )
        if resolved is None:
            continue
        task_id, live_status = resolved

        if live_status != snapshot_status:
            violations.append(
                {
                    'op_id': op.get('id'),
                    'task_id': task_id,
                    'snapshot_status': snapshot_status,
                    'live_status': live_status,
                }
            )

    return violations


def _check_flag_counter_completeness(
    report_stats: dict,
    prior_reports: list[StageReport],
) -> dict:
    """Compare ``report.stats['stage1_flags_processed']`` against Stage 1's truth.

    Stage 1 (memory_consolidator) emits ``StageReport.items_flagged`` — the
    definitive list of flags it raised for Stage 2 to process.  This guard
    compares that ground-truth count against whatever Stage 2 self-reported in
    ``stats['stage1_flags_processed']``.

    Pure stats-arithmetic — no taskmaster calls, no I/O.

    Args:
        report_stats: The ``StageReport.stats`` dict from Stage 2's run.
        prior_reports: Reports from earlier stages in this cycle.  When
            non-empty, ``prior_reports[0]`` must be the Stage 1
            (``memory_consolidator``) report — guarded by a stage-identity
            check to avoid a silent wrong-baseline comparison if the pipeline
            is ever reordered.  When empty, no baseline is available and the
            function returns ``mismatch=False`` unconditionally.

    Returns:
        ``{'expected': int, 'reported': int, 'mismatch': bool}``
        ``mismatch`` is ``True`` only when a Stage 1 baseline exists and
        ``reported != expected``.
    """
    reported = report_stats.get('stage1_flags_processed', 0)
    if not prior_reports:
        return {'expected': 0, 'reported': reported, 'mismatch': False}

    # Mirror the stage-identity guard used by the same-run dedup block above
    # (lines ~757-761) — only treat prior_reports[0] as a Stage 1 baseline
    # when it actually is Stage 1.  A wrong stage would produce a meaningless
    # expected count and silently clamp stats to garbage.
    if prior_reports[0].stage != StageId.memory_consolidator:
        return {'expected': 0, 'reported': reported, 'mismatch': False}

    expected = len(prior_reports[0].items_flagged)
    return {
        'expected': expected,
        'reported': reported,
        'mismatch': expected != reported,
    }


class Stage2FlagPartition(NamedTuple):
    """Partition result from :func:`_query_stage2_flags`.

    Using a NamedTuple makes both return values self-documenting at call sites
    and avoids positional-index surprises when the return shape evolves.
    Positional unpacking (``current, stale_ids = await _query_stage2_flags(...)``)
    continues to work unchanged since NamedTuple is a tuple subclass.

    Attributes:
        current: Full dict records whose ``metadata.run_id`` matches the active
            run.  Rendered to the Stage 2 LLM for FIX-C processing.
        stale_ids: ``id`` strings for records whose ``metadata.run_id`` is
            absent, empty, or mismatched.  Caller sweeps these via
            :func:`_sweep_stale_fixc_markers`.
    """

    current: list[dict]
    stale_ids: list[str]


async def _query_stage2_flags(
    memory_service,
    project_id: str,
    current_run_id: str,
) -> Stage2FlagPartition:
    """Query Mem0 for active Stage-2-destined flags and partition by run_id.

    Searches for memories with ``metadata.flag_for_stage2=true`` (the only
    supported convention — the ``stage1_flag_marker`` key was a never-shipped
    alias and is not checked here; see task-1139 reviewer note on dead code).
    Any other memories are discarded.

    Results are partitioned into two groups:

    * **current_flags** — full dict records whose ``metadata.run_id`` is
      present, non-empty, and matches ``current_run_id`` after ``str()``
      coercion.  These are rendered to the Stage 2 LLM for FIX-C processing.
    * **stale_marker_ids** — ``id`` strings only for records whose
      ``metadata.run_id`` is absent, empty, or does not match
      ``current_run_id``.  Markers missing or with an empty ``run_id`` are
      unconditionally classified as stale (legacy disposition: they pre-date
      the run_id producer contract and cannot be attributed to any specific
      run).  The caller is responsible for sweeping these via
      :func:`_sweep_stale_fixc_markers`.

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

    Returns ``([], [])`` on search failure (best-effort; logs WARNING).
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
        return Stage2FlagPartition([], [])

    if len(results) == 100:
        logger.warning(
            'reconciliation._query_stage2_flags: search returned limit=100 '
            'results — some markers may be beyond the top-N cutoff and will '
            'not be swept or rendered this cycle.  Follow-up: migrate to '
            'scroll_by_metadata for GC correctness.',
            extra={'project_id': project_id},
        )

    current_flags: list[dict] = []
    stale_marker_ids: list[str] = []
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
        # treated as absent and placed in the stale partition unconditionally.
        if meta.get('run_id') and str(meta['run_id']) == run_id_str:
            current_flags.append(flag_dict)
        else:
            stale_marker_ids.append(r.id)
    return Stage2FlagPartition(current_flags, stale_marker_ids)


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
_STAGE2_STALE_FIXC_SWEEP_SOURCE = 'stage2_stale_fixc_sweep'


async def _sweep_stale_fixc_markers(
    memory_service,
    project_id: str,
    stale_ids: list[str],
    run_id: str,
) -> int:
    """Delete stale fixc markers in parallel and return the count of successful deletes.

    Issues parallel ``delete_memory`` calls via ``asyncio.gather`` with
    ``return_exceptions=True`` (mirrors :func:`_track_flag_persistence` /
    :func:`_write_escalation_markers`).  Individual delete failures log WARNING
    and are excluded from the returned count — best-effort contract.

    Args:
        memory_service: The fused-memory service (must support ``delete_memory``).
        project_id: Project scope for the delete calls.
        stale_ids: List of Mem0 memory IDs to delete.  Empty list → returns 0
            immediately without issuing any calls.
        run_id: Current reconciliation run identifier used as ``causation_id``
            so the audit journal traces each delete back to the responsible cycle.

    Returns:
        Number of deletes that completed without raising (0 if *stale_ids* is empty).
    """
    if not stale_ids:
        return 0

    results = await asyncio.gather(
        *(
            memory_service.delete_memory(
                memory_id=mid,
                store='mem0',
                project_id=project_id,
                causation_id=run_id,
                _source=_STAGE2_STALE_FIXC_SWEEP_SOURCE,
            )
            for mid in stale_ids
        ),
        return_exceptions=True,
    )

    success_count = 0
    for mid, result in zip(stale_ids, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                'reconciliation._sweep_stale_fixc_markers: delete failed for memory_id=%s; not counted',
                mid,
                extra={'project_id': project_id, 'memory_id': mid, 'run_id': run_id},
            )
        else:
            success_count += 1
    return success_count


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
    Manual GC is acceptable — see design decision in plan.json.

    Note (task 1256): this function receives only *surviving_ids* — flags whose
    ``metadata.run_id`` differs from the active run are partitioned out by
    :func:`_query_stage2_flags` upstream; :func:`_sweep_stale_fixc_markers` then
    deletes them from Mem0 (independently of this call).  The counter therefore
    no longer observes Stage 2 delete failures from previous cycles; it only
    counts Stage 1 re-flags that survive within the current cycle's ``run_id``.
    """
    if not flag_ids:
        return {}

    # ── count phase (parallel) ───────────────────────────────────────────────
    count_results = await asyncio.gather(
        *(
            memory_service.count_memories_by_metadata(
                project_id=project_id,
                filters={'source': _STAGE2_PERSISTENCE_MARKER_SOURCE, 'flag_id': fid},
            )
            for fid in flag_ids
        ),
        return_exceptions=True,
    )
    prior_counts: dict[str, int] = {}
    for fid, result in zip(flag_ids, count_results, strict=True):
        if isinstance(result, BaseException):
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
    write_results = await asyncio.gather(
        *(
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
        ),
        return_exceptions=True,
    )
    for fid, result in zip(flag_ids, write_results, strict=True):
        if isinstance(result, BaseException):
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

    count_results = await asyncio.gather(
        *(
            memory_service.count_memories_by_metadata(
                project_id=project_id,
                filters={'source': _STAGE2_ESCALATION_MARKER_SOURCE, 'flag_id': fid},
            )
            for fid in flag_ids
        ),
        return_exceptions=True,
    )
    newly: list[str] = []
    already: list[str] = []
    for fid, result in zip(flag_ids, count_results, strict=True):
        if isinstance(result, BaseException):
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

    write_results = await asyncio.gather(
        *(
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
        ),
        return_exceptions=True,
    )
    for fid, result in zip(flag_ids, write_results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                'reconciliation._write_escalation_markers: add_memory failed for '
                'flag_id=%s; next cycle may re-escalate',
                fid,
                extra={'project_id': project_id, 'flag_id': fid},
            )


class TaskKnowledgeSync(BaseStage):
    """Stage 2: Reconcile tasks against memory, attach hints, fix inconsistencies."""

    # Remediation support — set by harness for second pass
    remediation_mode: bool = False

    # Active task tree — set by harness before run() (task 455)
    filtered_task_tree: FilteredTaskTree | None = None

    # Minimum number of tasks to proactively spot-check each run
    MIN_TASK_SAMPLE: int = 5

    # Current reconciliation run_id — set by run() so assemble_payload can use
    # it for stale-flag persistence markers (FIX D).
    # Sentinel None means run() has not yet been called; assemble_payload raises
    # loudly in that case so test authors are reminded to set this attribute.
    _current_run_id: str | None = None

    # Count of stale fixc markers swept by _sweep_stale_fixc_markers in the
    # current assemble_payload() call (task 1224).  Reset to 0 at the top of
    # run() so cross-invocation contamination is impossible.  Written to
    # report.stats['stale_fixc_markers_swept'] after super().run() returns.
    _stale_fixc_markers_swept: int = 0

    # Set to True by assemble_payload() when the autopilot_video contamination
    # guardrail fires (task IDs above AUTOPILOT_VIDEO_TASK_CEILING detected).
    # get_disallowed_tools() then adds DISALLOW_TASK_WRITES to the disallowed list
    # so the LLM cannot issue task writes even if it ignores the prompt guardrail.
    _contamination_detected: bool = False

    def get_system_prompt(self) -> str:
        return build_stage2_system_prompt(self.project_id)

    def get_disallowed_tools(self) -> list[str]:
        if self._contamination_detected:
            # Programmatic gate: when contamination is detected, block all task-
            # mutating tools so a non-compliant LLM cannot breach the guardrail.
            # This is defence-in-depth behind the prompt-level instruction.
            return STAGE2_DISALLOWED + DISALLOW_TASK_WRITES
        return STAGE2_DISALLOWED

    async def run(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
        run_id: str,
        model: str | None = None,
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
        self._stale_fixc_markers_swept = 0
        await self._maybe_queue_briefing_refresh_tasks(run_id=run_id)
        report = await super().run(events, watermark, prior_reports, run_id, model=model)

        # --- stale fixc marker sweep stat (task 1224) ---
        # _stale_fixc_markers_swept is set by assemble_payload() during the
        # super().run() call above.  Inject it into the report here so
        # downstream consumers (Stage 3 prompt, observability) can see how
        # many prior-cycle markers were swept (mirrors stage2_stage1_dups_suppressed).
        report.stats['stale_fixc_markers_swept'] = self._stale_fixc_markers_swept

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

        return report

    async def _apply_post_flight_guards(
        self,
        report: StageReport,
        prior_reports: list[StageReport],
        run_id: str,
    ) -> None:
        """Apply four orthogonal post-flight integrity guards to *report*.

        Each guard reads the write-journal op stream and/or calls
        ``taskmaster.get_task()`` to verify what the Stage 2 LLM actually did,
        then mutates ``report.stats`` in place to reflect the true picture.

        Guards run whenever ``super().run()`` returns (success or failure
        report), so partial-run artefacts are still classified correctly.
        If ``super().run()`` raises rather than returning, guards do not fire.

        Degrades gracefully when ``self.journal`` or
        ``self.journal.write_journal`` is ``None`` — Guards 1-3 skip (no ops
        available), Guard 4 still fires (pure stats arithmetic).

        Args:
            report: The ``StageReport`` returned by ``super().run()``.
                Mutated in place.
            prior_reports: Stage reports for all earlier stages in this cycle.
                Guard 4 reads ``prior_reports[0].items_flagged``.
            run_id: Current reconciliation run identifier.
        """
        # Derive agent_id from stage_id so it stays in sync with base.py:125.
        _stage_agent_id = f'recon-stage-{self.stage_id.value}'

        # Fetch write_journal ops once; share across Guards 1-3.
        ops: list[dict] = []
        if self.journal is not None and self.journal.write_journal is not None:
            try:
                ops = await self.journal.write_journal.get_ops_by_causation(run_id)
            except Exception:
                logger.warning(
                    'reconciliation._apply_post_flight_guards: '
                    'get_ops_by_causation failed for run_id=%s; '
                    'skipping Guards 1-3',
                    run_id,
                )
                ops = []

        # Pre-fetch all unique task_ids referenced by stage-2 ops concurrently,
        # building a shared status_cache so Guards 1-3 avoid N+1 get_task calls.
        status_cache: dict[str, str] | None = None
        if ops and self.taskmaster and self.project_root:
            task_ids: set[str] = set()
            for op in ops:
                if op.get('agent_id') != _stage_agent_id:
                    continue
                params_raw = op.get('params') or '{}'
                try:
                    params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
                except (json.JSONDecodeError, TypeError):
                    continue
                operation = op.get('operation')
                if operation in ('update_task', 'set_task_status'):
                    tid = str(params.get('task_id', '')).strip()
                    if tid:
                        task_ids.add(tid)
                elif operation == 'add_memory':
                    meta = params.get('metadata') or {}
                    if isinstance(meta, dict):
                        tid = str(meta.get('task_id', '')).strip()
                        if tid:
                            task_ids.add(tid)

            if task_ids:
                task_id_list = list(task_ids)
                fetch_results = await asyncio.gather(
                    *(self.taskmaster.get_task(tid, self.project_root) for tid in task_id_list),
                    return_exceptions=True,
                )
                status_cache = {}
                for tid, result in zip(task_id_list, fetch_results, strict=True):
                    if isinstance(result, BaseException):
                        logger.warning(
                            'reconciliation._apply_post_flight_guards: '
                            'get_task failed for task_id=%s during cache build; '
                            'guards will skip ops on this task',
                            tid,
                        )
                        continue  # omit entry; helpers detect missing key -> skip
                    if not isinstance(result, dict):
                        continue  # non-dict result; omit entry so helpers skip
                    extracted = _extract_status(result)
                    if extracted == 'unknown':
                        continue  # unresolvable status; omit entry so helpers skip
                    status_cache[tid] = extracted

        # Guard 1 — terminal-state pre-check
        if self.taskmaster and self.project_root:
            terminal_violations = await _classify_terminal_state_violations(
                ops, self.taskmaster, self.project_root, _stage_agent_id, status_cache
            )
            if terminal_violations:
                report.stats['not_applicable_count'] = report.stats.get(
                    'not_applicable_count', 0
                ) + len(terminal_violations)
                # Decrement inflated success counters (clamped at 0)
                report.stats['tasks_modified'] = max(
                    0,
                    report.stats.get('tasks_modified', 0) - len(terminal_violations),
                )
                for v in terminal_violations:
                    logger.info(
                        'reconciliation.skipped_done_task',
                        extra={
                            'run_id': run_id,
                            'project_id': self.project_id,
                            'task_id': v['task_id'],
                            'reason': v['reason'],
                        },
                    )

        # Guard 2 — stall-guard freshness gate
        if self.taskmaster and self.project_root:
            freshness_violations = await _check_stall_guard_freshness(
                ops, self.taskmaster, self.project_root, _stage_agent_id, status_cache
            )
            if freshness_violations:
                report.stats['stall_guard_freshness_violations'] = report.stats.get(
                    'stall_guard_freshness_violations', 0
                ) + len(freshness_violations)
                for v in freshness_violations:
                    logger.warning(
                        'reconciliation.stall_guard_freshness_violation',
                        extra={
                            'run_id': run_id,
                            'project_id': self.project_id,
                            'task_id': v['task_id'],
                            'snapshot_status': v['snapshot_status'],
                            'live_status': v['live_status'],
                        },
                    )

        # Guard 3 — post-action set_task_status verification
        if self.taskmaster and self.project_root:
            sts_mismatches = await _verify_set_task_status_post_action(
                ops, self.taskmaster, self.project_root, _stage_agent_id, status_cache
            )
            if sts_mismatches:
                report.stats['set_task_status_post_action_mismatches'] = report.stats.get(
                    'set_task_status_post_action_mismatches', 0
                ) + len(sts_mismatches)
                report.stats['tasks_modified'] = max(
                    0,
                    report.stats.get('tasks_modified', 0) - len(sts_mismatches),
                )
                for m in sts_mismatches:
                    logger.warning(
                        'reconciliation.set_task_status_post_action_mismatch',
                        extra={
                            'run_id': run_id,
                            'project_id': self.project_id,
                            'task_id': m['task_id'],
                            'target_status': m['target_status'],
                            'live_status': m['live_status'],
                        },
                    )

        # Guard 4 — flag-counter completeness (pure stats arithmetic, no I/O)
        flag_check = _check_flag_counter_completeness(report.stats, prior_reports)
        if flag_check['mismatch']:
            logger.warning(
                'reconciliation.stage1_flags_processed_mismatch',
                extra={
                    'run_id': run_id,
                    'project_id': self.project_id,
                    'expected': flag_check['expected'],
                    'reported': flag_check['reported'],
                },
            )
            # Clamp to truth so downstream verifiers see the real picture.
            report.stats['stage1_flags_processed'] = flag_check['expected']

    async def _maybe_queue_briefing_refresh_tasks(self, run_id: str = '') -> None:
        """Best-effort: queue 'Refresh briefing' tasks for each briefing-known-gaps mismatch.

        Silently skips if project_root or taskmaster is absent, or if project_id
        is not in ``_BRIEFING_REFRESH_PROJECT_ALLOWLIST`` (reify-specific feature).
        Any exception is caught and logged as a WARNING so a broken script can
        never abort Stage 2.
        """
        if not self.project_root or not self.taskmaster:
            return
        if self.project_id not in _BRIEFING_REFRESH_PROJECT_ALLOWLIST:
            return
        try:
            mismatches = await _run_briefing_known_gaps_script(self.project_root)
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
                self.project_root,
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
        # its docstring), so stale_marker_ids becomes non-empty even when no
        # flags are active.  _sweep_stale_fixc_markers (called unconditionally
        # whenever stale_marker_ids is non-empty) would then mass-delete all
        # fixc markers, corrupting the marker store with causation_id=''.
        # Raising here short-circuits before any filter_task_tree / Mem0 /
        # Taskmaster I/O, ensuring _track_flag_persistence and
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

        # Defence-in-depth: detect contamination early so get_disallowed_tools() can
        # block task-mutating tools for the remainder of this run.  The prompt-side
        # guardrail fires on the LLM's honour; the tool-block is the programmatic gate.
        if self.project_id == AUTOPILOT_VIDEO_PROJECT_ID:
            all_tasks = itertools.chain(
                filtered.active_tasks,
                filtered.done_tasks,
                filtered.cancelled_tasks,
            )
            excessive_ids = excessive_autopilot_video_ids(all_tasks)
            if excessive_ids:
                self._contamination_detected = True
                logger.warning(
                    'reconciliation.stage2_contamination_guard_fires '
                    'project_id=%s task_ids_above_ceiling=%s ceiling=%d — '
                    'verify this is cross-project contamination, not autopilot_video '
                    'growth past the ceiling constant; task-mutating tools will be '
                    'blocked for this run via get_disallowed_tools()',
                    self.project_id,
                    excessive_ids,
                    AUTOPILOT_VIDEO_TASK_CEILING,
                )

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
            self.project_root,
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

        # Compute the hint-attention section: active tasks whose memory_hints
        # need conversion from legacy list format (or are missing entirely).
        # Gated on `if not self.remediation_mode:` — mirrors proactive_sample_section
        # (above) because hint conversion is a general-sync activity, not a Stage 1
        # remediation task. Rendered conditionally within that gate — omitted on the
        # steady-state case where every active task already has valid dict-format hints.
        hint_conversion_section = ''
        if not self.remediation_mode:
            # Slice to the same window rendered by format_filtered_task_tree
            # (slice-then-filter): parity holds under the MAX_ACTIVE_TASKS_RENDERED
            # cap but NOT under format_filtered_task_tree's max_chars clamp — when
            # that secondary clamp fires, a few tail-position tasks may appear in
            # the hint section but be absent from the rendered tree.
            visible_active = filtered.active_tasks[:MAX_ACTIVE_TASKS_RENDERED]
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
        # _query_stage2_flags is best-effort: search failures yield ([], []) internally.
        # Returns (current_flags, stale_marker_ids): stale partition contains markers
        # whose metadata.run_id does not match the current run (or is absent — legacy
        # markers pre-dating the run_id producer contract).  Stale markers are swept
        # below by _sweep_stale_fixc_markers so they are never rendered to the LLM.
        run_id_for_markers = self._current_run_id
        active_flags, stale_marker_ids = await _query_stage2_flags(
            self.memory, self.project_id, run_id_for_markers
        )

        # SCOPE ADDITION (task 1139): apply the known-bug-1139 scope filter to
        # the active-query path ONLY.  Stage 1's structured-output flags are
        # intentionally emitted by the LLM and must not be suppressed here.
        surviving = [f for f in active_flags if not _should_skip_known_bug_1139_flag(f)]

        # FIX D — stale-flag persistence tracking.
        # Track how many cycles each surviving flag has survived without being
        # deleted.  Best-effort: _track_flag_persistence degrades gracefully.
        # Note (task 1256): :func:`_query_stage2_flags` partitions flags by
        # metadata.run_id, routing prior-cycle residue to stale_marker_ids
        # (not active_flags); :func:`_sweep_stale_fixc_markers` (called below)
        # then deletes them from Mem0.  FIX D therefore only fires on Stage 1
        # re-flags surviving within the current run_id.
        surviving_ids = [f['id'] for f in surviving]

        # Sweep stale markers (prior-cycle residue) in parallel.  Best-effort:
        # individual failures log WARNING but are not re-raised.  The count is
        # stored on the instance for reporting in run() via stats dict.
        self._stale_fixc_markers_swept = await _sweep_stale_fixc_markers(
            self.memory, self.project_id, stale_marker_ids, run_id_for_markers
        )
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
        # surviving Mem0 active-query results.
        combined_flags: list[dict] = list(stage1_report.items_flagged if stage1_report else [])
        for f in surviving:
            combined_flags.append(
                {
                    '_source': 'mem0_active_query',
                    'flag_id': f['id'],
                    'task_id': f['task_id'],
                    'content': f['content'],
                }
            )

        flagged_text = _format_flagged(combined_flags, run_stage='stage2')

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

        # Step 5 in the Your Task block below ("append=False for hint conversion")
        # is grounded in Mem0 memory 0b0eeb8d (old-wins semantics for list-format
        # hints under append=True).  The memory id is kept here rather than in
        # the prompt string so the LLM is not burdened with an opaque reference
        # it cannot look up, and the traceability survives prompt rewording.
        return f"""## Stage 2: Task-Knowledge Sync
## Project: {self.project_id}

{remediation_note}### Stage 1 Report Summary
{_format_report(stage1_report)}

### Stage 1 Flagged Items (Task-Relevant)
{flagged_text}
{stale_section}{known_projects_section}
{format_filtered_task_tree(filtered)}

### Recently Completed Tasks
{recently_completed_text}
{provenance_section}{proactive_sample_section}{hint_conversion_section}

## Your Task
Reconcile task state against memory:
1. For completed tasks: verify knowledge was captured. If sparse, search for related memories \
to check context, then write appropriate memories.
2. For tasks whose assumptions were invalidated by Stage 1 findings: modify, re-scope, or \
delete tasks. Update dependent tasks.
3. For AI-generated tasks: cross-reference against knowledge graph for factual consistency.
4. Attach memory_hints to tasks that would benefit from knowledge context at execution time. \
Use entity references + semantic queries, NOT inline content.
5. For tasks listed in **Tasks Needing Memory Hint Attention**: use read-modify-write with \
`append=False` when writing memory_hints — Stage 2's default `append=True` merge silently \
discards legacy list-format hints under old-wins semantics.
6. Proactively review the **Proactive Task Sample** regardless of Stage 1 findings: check \
in-progress tasks for completion knowledge to capture, blocked tasks for unblock conditions \
that may now be met, and done tasks for missing knowledge capture.
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

    def get_system_prompt(self) -> str:
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

        return STAGE3_SYSTEM_PROMPT

    def get_disallowed_tools(self) -> list[str]:
        return STAGE3_DISALLOWED

    def get_report_schema(self) -> dict:
        return STAGE3_REPORT_SCHEMA

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
    project_root: str | None,
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
    project_root: str,
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
    as a conversion target so the LLM uses read-modify-write with ``append=False``.
    """
    metadata = task.get('metadata')
    task_hints = metadata.get('memory_hints') if isinstance(metadata, dict) else None
    if isinstance(task_hints, list):
        return True
    return not task_hints


async def _run_briefing_known_gaps_script(project_root: str) -> list[dict] | None:
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
    project_root: str,
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
        except Exception:
            logger.warning(
                'briefing_refresh_add_task_failed',
                exc_info=True,
                extra={'project_root': project_root, 'task_id': task_id},
            )
            failed.append(task_id)

    return {'created': created, 'skipped': skipped, 'failed': failed}
