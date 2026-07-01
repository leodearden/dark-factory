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
    STAGE2_DISALLOWED,
    STAGE3_DISALLOWED,
    STAGE3_REPORT_SCHEMA,
    build_summary_nonce_section,
    generate_summary_nonce,
)
from fused_memory.reconciliation.flag_dedup import (
    compute_flag_signature,
    filter_blocked_snapshot_findings,
)
from fused_memory.reconciliation.prompts import (
    _STAGE2_PROJECT_ID_GUIDELINE,
    _STAGE3_PROJECT_ID_GUIDELINE,
)
from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.summary_pool import (
    enforce_summary_pool_cap,
    pretrim_summary_pool,
)
from fused_memory.reconciliation.task_filter import (
    FilteredTaskTree,
    detect_task_dump_contamination,
    filter_task_tree,
    format_task_list,
    id_key,
    render_active_section,
)
from fused_memory.services.live_workflow_detector import (
    detect_live_workflow,
    is_workflow_live_for_task,
)
from fused_memory.services.orchestrator_detector import is_orchestrator_live_for

logger = logging.getLogger(__name__)


def _extract_status(task_data: dict) -> str:
    """Extract status from a Taskmaster get_task response dict.

    **Sibling copies** — keep in sync if the get_task response shape ever changes:

    * ``middleware/task_interceptor._extract_status`` (~line 3292) — canonical source
    * ``reconciliation/flag_dedup._extract_terminal_status`` — same logic with an
      additional non-dict guard (returns ``'unknown'`` for non-dict input)
    """
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


async def _classify_live_workflow_status_writes(
    ops: list[dict],
    project_root: str,
    agent_id: str,
    *,
    now=None,
) -> list[dict]:
    """Classify write_journal ops where Stage 2 wrote set_task_status on a live-workflow task.

    For every ``set_task_status`` op authored by *agent_id*, parses the params
    for task_id, then calls :func:`is_workflow_live_for_task`.  When the task is
    live, returns a violation record ``{op_id, task_id, reason: 'live_workflow_status_write'}``.

    This is a post-flight observation guard (defence-in-depth): the write already
    landed (the LLM issued it via MCP), so this helper cannot undo it.  Callers
    are expected to:

    - increment ``stats['live_workflow_status_writes']`` by ``len(violations)``
    - decrement ``stats['tasks_modified']`` by ``len(violations)`` (clamped at 0)

    Filter rules (mirror :func:`_classify_terminal_state_violations`):

    * Only ops where ``op['agent_id'] == agent_id`` are inspected.
    * Only ops where ``op['operation'] == 'set_task_status'`` are inspected.
    * Detector exceptions are swallowed (treat as not-live — fail toward logging
      the op as a legitimate action rather than flagging it spuriously).

    Args:
        ops: Write-journal op dicts (``layer=='write_op'``).
        project_root: Absolute path to the Taskmaster project directory.
        agent_id: The agent_id string to filter ops on (derived from stage_id).
        now: Injectable datetime for deterministic tests (forwarded to detector).

    Returns:
        List of ``{'op_id', 'task_id', 'reason': 'live_workflow_status_write'}``
        dicts, one per qualifying violation.  Empty list when no violations found.
    """
    violations: list[dict] = []
    for op in ops:
        if op.get('agent_id') != agent_id:
            continue
        if op.get('operation') != 'set_task_status':
            continue

        params_raw = op.get('params') or '{}'
        try:
            params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                'reconciliation._classify_live_workflow_status_writes: '
                'failed to parse params JSON for op_id=%s; skipping',
                op.get('id'),
            )
            continue
        task_id = str(params.get('task_id', '')).strip()
        if not task_id:
            continue

        try:
            kwargs = {} if now is None else {'now': now}
            live = is_workflow_live_for_task(task_id, project_root, **kwargs)
        except Exception:
            logger.warning(
                'reconciliation._classify_live_workflow_status_writes: '
                'detector error for task_id=%s; treating as not-live',
                task_id,
            )
            live = False

        if live:
            violations.append(
                {
                    'op_id': op.get('id'),
                    'task_id': task_id,
                    'reason': 'live_workflow_status_write',
                }
            )

    return violations


def _check_flag_counter_completeness(
    report_stats: dict,
    prior_reports: list[StageReport],
) -> dict:
    """Compare ``report.stats['stage1_analytical_findings_processed']`` against Stage 1's truth.

    Stage 1 (memory_consolidator) emits ``StageReport.items_flagged`` — the
    definitive list of structured analytical flags it raised for Stage 2 to
    review.  This guard compares that ground-truth count against whatever
    Stage 2 self-reported in ``stats['stage1_analytical_findings_processed']``
    (the count of Stage 1 flagged_items that Stage 2 actually reviewed).

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
    reported = report_stats.get('stage1_analytical_findings_processed')
    if reported is None:
        # Backward-compat: in-flight or cached Stage 2 agents may still emit the
        # pre-rename key (task 1589).  Prompt and reader move together at deploy
        # time, but the fallback ensures no spurious mismatch during rollout when
        # an old-prompt agent is still running.
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


def _check_mem0_flag_counter_completeness(report_stats: dict) -> dict:
    """Compare ``report.stats['stage1_mem0_flags_processed']`` against the flag_deleted records.

    The Stage 2 prompt requires emitting one ``flag_deleted`` action record per FIX C
    deletion into ``stats['flag_deleted_records']`` — a list of
    ``{"action": "flag_deleted", "flag_id": ..., "reason": ...}`` dicts.
    This guard uses that list as the ground-truth count so an agent that under- or
    over-reports the Mem0 counter is caught and clamped, mirroring the analytical guard
    in :func:`_check_flag_counter_completeness`.

    Pure stats-arithmetic — no taskmaster calls, no I/O.

    Args:
        report_stats: The ``StageReport.stats`` dict from Stage 2's run.

    Returns:
        ``{'expected': int, 'reported': int, 'mismatch': bool}``
        ``expected`` is the count of ``{"action": "flag_deleted", ...}`` dicts in
        ``stats['flag_deleted_records']`` (defaults to 0 when the key is absent or
        the list is empty).  ``mismatch`` is ``True`` only when
        ``expected != reported``.
    """
    records = report_stats.get('flag_deleted_records', [])
    expected = sum(
        1 for r in (records if isinstance(records, list) else [])
        if isinstance(r, dict) and r.get('action') == 'flag_deleted'
    )
    reported = report_stats.get('stage1_mem0_flags_processed', 0)
    return {
        'expected': expected,
        'reported': reported,
        'mismatch': expected != reported,
    }


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
            :func:`_query_stage2_flags` when this bucket is non-empty.  Caller
            sweeps these via :func:`_sweep_stale_fixc_markers`.
        stale_mismatched_run_id_ids: ``id`` strings for records whose
            ``metadata.run_id`` is present and truthy but does not match the
            current ``run_id`` AND whose ``created_at`` is out of the run
            window.  These are normal prior-cycle residue.  Caller sweeps these
            via :func:`_sweep_stale_fixc_markers`.
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

    Both stale buckets must be swept by the caller via
    :func:`_sweep_stale_fixc_markers`.

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

# Stage 2 per-cycle summary pool cap and related constants.
# Every per-cycle summary add_memory call is tagged recon_pool='stage2_cycle_summary'
# (producer contract: Stage 2 prompt).  After the LLM writes its summary,
# _enforce_stage2_summary_pool_cap trims the pool to at most this many members by
# deleting the OLDEST entries — deterministically via Qdrant scroll, NOT semantic search.
_STAGE2_CYCLE_SUMMARY_RECON_POOL = 'stage2_cycle_summary'
STAGE2_CYCLE_SUMMARY_POOL_CAP: int = 2
_STAGE2_CYCLE_SUMMARY_TRIM_SOURCE = 'stage2_cycle_summary_trim'

# Audit tag for _repair_stage2_summary_stage_metadata's delete+re-add repair
# (task 1963) — heals the intermittent LLM-compliance failure where the Stage
# 2 cycle_summary write omits metadata.stage='task_knowledge_sync' (or, more
# broadly, kind='cycle_summary'), which makes the downstream triple-filter
# count_memories_by_metadata check in _verify_stage2_summary_written falsely
# report 0.
_STAGE2_SUMMARY_STAGE_REPAIR_SOURCE = 'stage2_summary_stage_repair'

# Audit tag for _reconstruct_stage2_summary's retroactive reconstruction write
# (task 1964) — closes the FULLY-ABSENT gap: when the Stage 2 per-cycle
# summary pool has ZERO members for a run_id (LLM crash before write, a
# silent Mem0 dedup drop, or a wrong/absent run_id), neither
# _verify_stage2_summary_written (observe-only) nor
# _repair_stage2_summary_stage_metadata (heals mislabeled-but-present
# summaries; its enumeration returns [] when nothing exists) writes a
# discoverable summary. This helper writes ONE dedup-resilient placeholder,
# automating the manual reconstruction performed for run 6467daca.
_STAGE2_SUMMARY_RECONSTRUCTION_SOURCE = 'stage2_summary_reconstruction'

# Age-based GC for orphaned stage1_flag_marker records (task 1944).
# flag_dedup._write_and_confirm_marker rewrites a stage1_flag_marker with a
# fresh created_at on every dedup HIT (a finding that keeps recurring); a
# finding that stops recurring leaves its marker orphaned with no other
# collector (documented as the "orphan-growth caveat" in flag_dedup.py,
# task-1670).  14 days is a config default, not a numeric tolerance asserted
# against live data: reify recon runs multiple cycles/day, so 14 days
# untouched means the finding has been dead across dozens of cycles.  The
# cost of a false GC is bounded to exactly one re-escalation — the next
# occurrence is simply a MISS that writes a fresh marker — within the
# existing best-effort-replacement tolerance already documented there.
STAGE1_FLAG_MARKER_MAX_AGE_DAYS: int = 14
_STAGE1_FLAG_MARKER_GC_SWEEP_SOURCE = 'stage1_flag_marker_gc_sweep'


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


async def _sweep_stale_flag_markers(
    memory_service,
    project_id: str,
    run_id: str,
    *,
    max_age_days: int = STAGE1_FLAG_MARKER_MAX_AGE_DAYS,
    now: datetime | None = None,
    scroll_limit: int = 1000,
) -> int:
    """Age-GC orphaned ``stage1_flag_marker`` Mem0 records (task 1944).

    ``stage1_flag_marker`` records are only garbage-collected today on the
    dedup HIT path (``flag_dedup._write_and_confirm_marker`` deletes priors
    when the SAME signature re-flags).  A finding that stops recurring leaves
    its marker orphaned — Stage 2's flag sweeps only touch records carrying
    ``metadata.flag_for_stage2=true``, which stage1_flag_markers never do, and
    ``scripts/sweep_orphan_flag_markers.py`` only deletes markers *missing*
    ``kind='stage1_flag_marker'`` (a disjoint, legacy-orphan concern). This
    helper closes that gap by enumerating the whole pool and deleting members
    whose ``created_at`` is strictly older than ``now - max_age_days``.

    Enumerates deterministically via ``memory_service.get_memories_by_metadata``
    (Qdrant payload-filter scroll) — NEVER semantic search, which silently
    drops low-similarity rows and is unsuitable for exhaustive GC.  Members
    with a missing or unparseable ``created_at`` are KEPT (never deleted),
    mirroring ``summary_pool._sort_key``'s sort-last / prefer-keep posture.

    Deletes are issued best-effort in parallel via ``asyncio.gather`` with
    ``return_exceptions=True`` (mirrors :func:`_sweep_stale_fixc_markers`):
    individual failures log WARNING and are excluded from the returned count.

    Args:
        memory_service: Service with ``get_memories_by_metadata`` and
            ``delete_memory``.
        project_id: Project scope for enumeration and delete calls.
        run_id: Current reconciliation run identifier used as ``causation_id``
            in the audit journal.
        max_age_days: Staleness cutoff in days (default
            ``STAGE1_FLAG_MARKER_MAX_AGE_DAYS`` == 14).
        now: Reference "current time" for the cutoff calculation. Defaults to
            ``datetime.now(UTC)``; tests inject a fixed value.
        scroll_limit: Max records to enumerate in one scroll (default 1000).

    Returns:
        Number of memories successfully deleted (0 if nothing is stale, or
        on enumeration failure).
    """
    try:
        members = await memory_service.get_memories_by_metadata(
            project_id=project_id,
            filters={'source': 'stage1_flag_marker'},
            limit=scroll_limit,
        )
    except Exception:
        logger.warning(
            'reconciliation._sweep_stale_flag_markers: '
            'get_memories_by_metadata failed for project_id=%s; skipping sweep',
            project_id,
            extra={'project_id': project_id, 'run_id': run_id},
        )
        return 0

    if len(members) >= scroll_limit:
        logger.warning(
            'reconciliation._sweep_stale_flag_markers: enumerated %d of scroll_limit=%d '
            'stage1_flag_marker records — scroll cap reached; older stale markers may '
            'remain uncollected this cycle; re-run with a higher scroll_limit.',
            len(members), scroll_limit,
            extra={'project_id': project_id, 'run_id': run_id},
        )

    if not members:
        return 0

    cutoff = _assume_utc(now or datetime.now(UTC)) - timedelta(days=max_age_days)

    stale_ids: list[str] = []
    for member in members:
        mid = member.get('id')
        if not mid:
            continue
        raw = member.get('created_at')
        if raw is None:
            continue
        try:
            created_at = _assume_utc(datetime.fromisoformat(raw))
        except (ValueError, TypeError):
            continue
        if created_at < cutoff:
            stale_ids.append(mid)

    if not stale_ids:
        return 0

    results = await asyncio.gather(
        *(
            memory_service.delete_memory(
                memory_id=mid,
                store='mem0',
                project_id=project_id,
                causation_id=run_id,
                _source=_STAGE1_FLAG_MARKER_GC_SWEEP_SOURCE,
            )
            for mid in stale_ids
        ),
        return_exceptions=True,
    )

    success_count = 0
    for mid, result in zip(stale_ids, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                'reconciliation._sweep_stale_flag_markers: delete failed for memory_id=%s; not counted',
                mid,
                extra={'project_id': project_id, 'memory_id': mid, 'run_id': run_id},
            )
        else:
            success_count += 1
    return success_count


async def _enforce_stage2_summary_pool_cap(
    memory_service,
    project_id: str,
    run_id: str,
    cap: int = STAGE2_CYCLE_SUMMARY_POOL_CAP,
) -> int:
    """Trim the stage2_cycle_summary pool to at most *cap* members (default 2).

    Thin delegator to the generic ``reconciliation.summary_pool.enforce_summary_pool_cap``
    core (task 1942 extraction) — behavior is unchanged from the original
    Stage-2-only implementation. Kept as a distinct public name/signature so
    the ~30 existing Stage 2 pool-cap tests (and any other callers) are
    unaffected by the extraction.

    Args:
        memory_service: Service with ``get_memories_by_metadata`` and
            ``delete_memory``.
        project_id: Project scope for enumeration and delete calls.
        run_id: Current reconciliation run identifier used as ``causation_id``
            in the audit journal.
        cap: Maximum pool size to enforce (default
            ``STAGE2_CYCLE_SUMMARY_POOL_CAP == 2``).

    Returns:
        Number of memories successfully deleted (0 if pool <= cap or on
        enumeration failure).
    """
    return await enforce_summary_pool_cap(
        memory_service,
        project_id,
        run_id,
        recon_pool=_STAGE2_CYCLE_SUMMARY_RECON_POOL,
        trim_source=_STAGE2_CYCLE_SUMMARY_TRIM_SOURCE,
        cap=cap,
    )


async def _pretrim_stage2_summary_pool(
    memory_service,
    project_id: str,
    run_id: str,
    cap: int = STAGE2_CYCLE_SUMMARY_POOL_CAP,
) -> int:
    """Pre-trim the stage2_cycle_summary pool to cap-1, reserving one slot.

    Thin delegator to the generic ``reconciliation.summary_pool.pretrim_summary_pool``
    core (task 1942 extraction) — behavior is unchanged from the original
    Stage-2-only implementation: delegates with ``cap=max(cap - 1, 0)`` so the
    imminent agent write lands as the cap-th member and can never be a trim
    candidate (trim-then-write ordering, task 1831).

    Must be called BEFORE ``super().run()`` (the agent write).  Post-write
    pool size transiently reaches cap; it is bounded back to cap on the next
    cycle's pre-trim — no post-write trim is needed.

    Args:
        memory_service: Forwarded to :func:`_enforce_stage2_summary_pool_cap`.
        project_id: Project scope.
        run_id: Current reconciliation run identifier (audit journal).
        cap: Logical pool cap (default ``STAGE2_CYCLE_SUMMARY_POOL_CAP``).
            Actual trim target is ``max(cap - 1, 0)``.

    Returns:
        Number of memories successfully deleted (0 if pool is already at
        or below cap-1, or on enumeration failure).
    """
    return await pretrim_summary_pool(
        memory_service,
        project_id,
        run_id,
        recon_pool=_STAGE2_CYCLE_SUMMARY_RECON_POOL,
        trim_source=_STAGE2_CYCLE_SUMMARY_TRIM_SOURCE,
        cap=cap,
    )


async def _verify_stage2_summary_written(
    memory_service,
    project_id: str,
    run_id: str,
) -> int:
    """Best-effort post-write check: count this run's cycle_summary in Mem0.

    Counts memories matching the triple filter
    ``{'kind':'cycle_summary','stage':'task_knowledge_sync','run_id':run_id}``
    via ``count_memories_by_metadata`` (deterministic Qdrant count, NOT semantic
    search).  Logs a WARNING when count==0 or when the count call raises.

    The Stage 2 prompt mandates the cycle_summary write unconditionally on every
    cycle path (full and remediation); there is no legitimate skip path.
    count==0 is always unexpected and indicates a genuine agent failure or
    write loss — the WARNING is intentional and not a false-positive alarm.

    Best-effort: never raises, never retries.  The LLM agent owns its own
    count-verify-and-retry path; Python's verify is a redundant observability
    signal surfaced via ``report.stats['stage2_cycle_summary_verified_count']``.

    Args:
        memory_service: Service with ``count_memories_by_metadata``.
        project_id: Project scope for the count call.
        run_id: Current reconciliation run identifier (used as filter key).

    Returns:
        Count returned by ``count_memories_by_metadata``, or 0 on failure.
    """
    try:
        count = await memory_service.count_memories_by_metadata(
            project_id=project_id,
            filters={
                'kind': 'cycle_summary',
                'stage': 'task_knowledge_sync',
                'run_id': run_id,
            },
        )
    except Exception:
        logger.warning(
            'reconciliation._verify_stage2_summary_written: '
            'count_memories_by_metadata failed for run_id=%s; treating as 0',
            run_id,
            extra={'project_id': project_id, 'run_id': run_id},
        )
        return 0
    if not count:
        logger.warning(
            'reconciliation._verify_stage2_summary_written: '
            'no cycle_summary found for run_id=%s after agent write',
            run_id,
            extra={'project_id': project_id, 'run_id': run_id},
        )
    return count


async def _repair_stage2_summary_stage_metadata(
    memory_service,
    project_id: str,
    run_id: str,
) -> int:
    """Repair Stage 2 cycle_summary writes missing the required stage identity.

    The Stage 2 per-cycle summary is written by the LLM agent inside
    ``super().run()`` via ``add_memory`` with metadata
    ``{kind:'cycle_summary', stage:'task_knowledge_sync', run_id, recon_pool:
    'stage2_cycle_summary'}``.  Intermittently the LLM omits one of the
    identity keys — the confirmed 2026-07-01 incident showed ``run_id`` and
    ``recon_pool`` present with only ``stage`` missing — which makes the
    downstream triple-filter ``count_memories_by_metadata({kind, stage,
    run_id})`` check in :func:`_verify_stage2_summary_written` falsely report
    0 ("Stage 2 summary missing").

    There is no in-place metadata-mutation primitive available: Mem0's
    ``update()`` rewrites the memory's TEXT only, not its payload metadata.
    The repair is therefore delete + re-add under corrected metadata — the
    same idiom used for this incident's data-level repair.

    Enumerates this run's ``stage2_cycle_summary`` pool members
    deterministically via ``memory_service.get_memories_by_metadata`` filtered
    on ``{recon_pool, run_id}`` (never semantic search — ``recon_pool`` and
    ``run_id`` are reliably present per the incident, so this scopes exactly
    to the current run's summary/summaries). For each member that does not
    already satisfy the full identity (``kind == 'cycle_summary'`` AND
    ``stage == 'task_knowledge_sync'``), its content is recovered from the
    Mem0 scroll payload's ``data`` key (the verbatim content under
    ``infer=False``) — falling back to a synthesized placeholder when absent
    — and re-added under the four canonical metadata keys.  The corrected
    copy is added BEFORE the broken original is deleted (add-before-delete):
    a crash between the two writes must never leave the run with zero
    correctly-tagged summaries, which is precisely the false-missing state
    this helper exists to eliminate.

    Best-effort: never raises. Enumeration failure logs a WARNING and returns
    0 (mirrors :func:`_sweep_stale_flag_markers`). A per-member add/delete
    failure logs a WARNING, excludes that member from the returned count, and
    does not abort the remaining members (mirrors :func:`_sweep_stale_fixc_markers`'s
    per-item isolation).

    Args:
        memory_service: Service with ``get_memories_by_metadata``,
            ``add_memory``, and ``delete_memory``.
        project_id: Project scope for enumeration and write calls.
        run_id: Current reconciliation run identifier — both the enumeration
            filter key and the ``causation_id`` for the repair writes.

    Returns:
        Number of members successfully repaired (0 if none are broken, none
        are found, or on enumeration failure).
    """
    try:
        members = await memory_service.get_memories_by_metadata(
            project_id=project_id,
            filters={
                'recon_pool': _STAGE2_CYCLE_SUMMARY_RECON_POOL,
                'run_id': run_id,
            },
        )
    except Exception:
        logger.warning(
            'reconciliation._repair_stage2_summary_stage_metadata: '
            'get_memories_by_metadata failed for run_id=%s; skipping repair',
            run_id,
            extra={'project_id': project_id, 'run_id': run_id},
        )
        return 0

    repaired = 0
    for member in members:
        metadata = member.get('metadata') or {}
        if metadata.get('stage') != 'task_knowledge_sync' or metadata.get('kind') != 'cycle_summary':
            mid = member.get('id')
            content = metadata.get('data') or (
                f'Stage 2 cycle summary (metadata-repaired) for run {run_id}'
            )
            corrected_metadata = {
                'kind': 'cycle_summary',
                'stage': 'task_knowledge_sync',
                'run_id': run_id,
                'recon_pool': _STAGE2_CYCLE_SUMMARY_RECON_POOL,
            }
            try:
                await memory_service.add_memory(
                    content=content,
                    category='observations_and_summaries',
                    project_id=project_id,
                    metadata=corrected_metadata,
                    causation_id=run_id,
                    _source=_STAGE2_SUMMARY_STAGE_REPAIR_SOURCE,
                )
                await memory_service.delete_memory(
                    memory_id=mid,
                    store='mem0',
                    project_id=project_id,
                    causation_id=run_id,
                    _source=_STAGE2_SUMMARY_STAGE_REPAIR_SOURCE,
                )
            except Exception:
                logger.warning(
                    'reconciliation._repair_stage2_summary_stage_metadata: '
                    'repair failed for memory_id=%s run_id=%s; not counted',
                    mid, run_id,
                    extra={'project_id': project_id, 'memory_id': mid, 'run_id': run_id},
                )
                continue
            repaired += 1

    return repaired


def _extract_response_memory_ids(response) -> list:
    """Defensively read ``memory_ids`` from an ``add_memory`` response.

    Production calls return :class:`~fused_memory.models.memory.AddMemoryResponse`
    (attribute access); tests in this module commonly mock ``add_memory`` with a
    plain ``{'memory_ids': [...]}`` dict. Supports both shapes so callers never
    need to know which one they were handed.

    Returns:
        The ``memory_ids`` list, or ``[]`` if absent/falsy on either shape.
    """
    if isinstance(response, dict):
        return response.get('memory_ids') or []
    return getattr(response, 'memory_ids', None) or []


def _build_stage2_reconstruction_content(run_id: str) -> str:
    """Build the retroactive-reconstruction placeholder content for *run_id*.

    Leads with a fresh ``generate_summary_nonce('STAGE2')`` line — the same
    CSPRNG dedup-defeat primitive the LLM per-cycle-summary path uses (task
    1572/1590) — so repeat calls (the one-shot retry in
    :func:`_reconstruct_stage2_summary`) never collide on Mem0's ~0.92
    cosine-similarity dedup threshold. The body is explicitly labeled a
    harness reconstruction and references *run_id* so a human (or another
    repair pass) can immediately identify it as synthetic, non-agent content.
    """
    nonce = generate_summary_nonce('STAGE2')
    return (
        f'{nonce}\n'
        f'Stage 2 cycle summary (retroactive reconstruction) for run {run_id} — '
        'original per-cycle summary absent after LLM write and metadata repair; '
        'reconstructed by harness self-heal.'
    )


async def _reconstruct_stage2_summary(
    memory_service,
    project_id: str,
    run_id: str,
) -> int:
    """Retroactively reconstruct a FULLY-ABSENT Stage 2 per-cycle summary.

    Closes the residual gap left by the existing verify/repair chain: when
    the ``stage2_cycle_summary`` pool has ZERO members for *run_id* —
    LLM crash/timeout before the write was ever sent, a silent Mem0 dedup
    no-op (``memory_ids=[]``), or a wrong/absent ``run_id`` — the task-1796
    verify (:func:`_verify_stage2_summary_written`) only logs a WARNING, and
    the task-1963 repair (:func:`_repair_stage2_summary_stage_metadata`)
    enumerates ``get_memories_by_metadata({recon_pool, run_id})``, finds
    nothing, and repairs nothing. Neither writes a discoverable summary.
    This helper is the automated form of the manual retroactive placeholder
    Stage 1 wrote for run 6467daca (mem0 memory 4a3a42d1).

    Writes ONE placeholder via ``add_memory`` tagged with the four canonical
    identity keys (``kind='cycle_summary'``, ``stage='task_knowledge_sync'``,
    ``run_id``, ``recon_pool='stage2_cycle_summary'``) plus
    ``reconstructed=True``, and ``_source=_STAGE2_SUMMARY_RECONSTRUCTION_SOURCE``
    so the write is auditable and distinguishable from both agent writes and
    task-1963 repairs. The content leads with a CSPRNG nonce (see
    :func:`_build_stage2_reconstruction_content`) to defeat Mem0's dedup.

    Args:
        memory_service: Service with ``add_memory``.
        project_id: Project scope for the write.
        run_id: Current reconciliation run identifier — both the metadata
            identity key and the ``causation_id`` for the write.

    Returns:
        1 if the response's ``memory_ids`` (see
        :func:`_extract_response_memory_ids`) is non-empty, else 0.
    """
    metadata = {
        'kind': 'cycle_summary',
        'stage': 'task_knowledge_sync',
        'run_id': run_id,
        'recon_pool': _STAGE2_CYCLE_SUMMARY_RECON_POOL,
        'reconstructed': True,
    }
    response = await memory_service.add_memory(
        content=_build_stage2_reconstruction_content(run_id),
        category='observations_and_summaries',
        project_id=project_id,
        metadata=metadata,
        causation_id=run_id,
        _source=_STAGE2_SUMMARY_RECONSTRUCTION_SOURCE,
    )
    if _extract_response_memory_ids(response):
        return 1
    return 0


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


def _render_live_workflow_section(
    tasks: list[dict],
    project_root: str,
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

    Args:
        tasks: Task dicts from the active/proactive-sample pool.  Only tasks
            with a parseable ``id`` are inspected (non-int ids are skipped).
        project_root: Absolute path to the project root, forwarded to the
            detector.
        now: Injectable reference time for deterministic tests.

    Returns:
        A Markdown section string (e.g. ``'### Live-Workflow Signals\\n...\\n'``),
        or ``''`` when no tasks are live.
    """
    if not project_root or not tasks:
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
    live_lines: list[str] = []

    for task in tasks:
        raw_id = task.get('id')
        if raw_id is None:
            continue
        task_id = str(raw_id)
        try:
            liveness = detect_live_workflow(task_id, project_root, **kwargs)
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

    # Current reconciliation run_id — set by run() so assemble_payload can use
    # it for stale-flag persistence markers (FIX D).
    # Sentinel None means run() has not yet been called; assemble_payload raises
    # loudly in that case so test authors are reminded to set this attribute.
    _current_run_id: str | None = None

    # Count of stale fixc markers swept by _sweep_stale_fixc_markers in the
    # current assemble_payload() call (task 1224).  Reset to 0 at the top of
    # run() so cross-invocation contamination is impossible.  Written to
    # report.stats['stale_fixc_markers_swept'] after super().run() returns.
    # Note (task-1369): same-run in-window markers with absent/mismatched run_id
    # are rescued to partition.current by the run-window guard in
    # _query_stage2_flags and therefore NEVER enter the stale buckets — this
    # counter reflects only genuine prior-cycle residue.
    _stale_fixc_markers_swept: int = 0

    # Count of Stage 2 markers with absent/empty metadata.run_id that were routed
    # to the stale_missing_run_id bucket in the current assemble_payload() call
    # (task 1257).  Non-zero indicates Stage 1 producer drift from a PRIOR cycle —
    # the LLM omitted the required run_id field.  Reset and injected via the same
    # four-touchpoint pattern as _stale_fixc_markers_swept.
    # Note (task-1369): same-cycle markers with absent run_id whose Mem0 created_at
    # is within the current run window are rescued to partition.current by
    # _query_stage2_flags and are NOT counted here.
    _stale_missing_run_id_markers: int = 0

    # Count of Stage 2 markers rescued to partition.current by the run-window guard
    # in _query_stage2_flags (task 1369).  Non-zero indicates Stage 1 producer drift
    # within the CURRENT cycle — the LLM omitted or mis-stamped metadata.run_id on a
    # flag it wrote during this run.  The marker was still surfaced to the Stage 2 LLM
    # (not swept), but the drift is recorded here for operator observability.  Reset
    # and injected via the same four-touchpoint pattern as _stale_fixc_markers_swept.
    _rescued_in_window_markers: int = 0

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
        self._stale_missing_run_id_markers = 0
        self._rescued_in_window_markers = 0
        await self._maybe_queue_briefing_refresh_tasks(run_id=run_id)
        # --- trim-then-write: pre-trim pool to cap-1 BEFORE agent writes (task 1831) ---
        # Runs unconditionally (full + remediation) so the pool is bounded every cycle.
        # Trim target is cap-1, reserving one slot for the imminent agent write so a
        # newly-written summary can never be the "oldest" trim candidate.  Best-effort.
        pretrimmed = await _pretrim_stage2_summary_pool(self.memory, self.project_id, run_id)
        report = await super().run(events, watermark, prior_reports, run_id, model=model)

        # --- stale fixc marker sweep stat (task 1224) ---
        # _stale_fixc_markers_swept is set by assemble_payload() during the
        # super().run() call above.  Inject it into the report here so
        # downstream consumers (Stage 3 prompt, observability) can see how
        # many prior-cycle markers were swept (mirrors stage2_stage1_dups_suppressed).
        report.stats['stale_fixc_markers_swept'] = self._stale_fixc_markers_swept
        # --- missing-run_id marker stat (task 1257) ---
        # Mirrors _stale_fixc_markers_swept; explicit zero is required so
        # downstream consumers never need .get(..., 0) fallbacks.
        report.stats['stale_missing_run_id_markers'] = self._stale_missing_run_id_markers
        # --- rescued-in-window marker stat (task-1369 amendment) ---
        # Non-zero when the run-window guard rescued same-cycle Stage-1 markers
        # whose run_id was omitted/mis-stamped.  They reached Stage 2 fine (not
        # swept), but the count is observable here for operator diagnostics.
        # Explicit zero so downstream consumers never need .get(..., 0) fallbacks.
        report.stats['rescued_in_window_markers'] = self._rescued_in_window_markers

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

        # --- cycle-summary stats (task 1831 trim-then-write) ---
        # pre-trim count was captured before super().run(); record it now alongside
        # the post-write verification count so both land in the same report.stats.
        report.stats['stage2_cycle_summary_pool_trimmed'] = pretrimmed
        verified_count = await _verify_stage2_summary_written(self.memory, self.project_id, run_id)

        # --- cycle-summary stage-metadata repair (task 1963) ---
        # verified_count==0 means the triple-filter found no cycle_summary for
        # this run — either the LLM genuinely failed to write one, or it wrote
        # one but omitted (part of) the identity metadata (the confirmed
        # 2026-07-01 incident: stage missing while kind/run_id/recon_pool were
        # present). Gate the repair on count==0 so the happy path (LLM
        # complied) costs zero extra Mem0 calls and carries zero
        # duplicate-write risk; when the summary is genuinely absent,
        # enumeration returns [] and repaired stays 0 (no false repair).
        # report.stats['stage2_cycle_summary_verified_count'] is set to the
        # POST-repair count so downstream consumers see the corrected value.
        repaired = 0
        if verified_count == 0:
            repaired = await _repair_stage2_summary_stage_metadata(self.memory, self.project_id, run_id)
            if repaired:
                verified_count = await _verify_stage2_summary_written(self.memory, self.project_id, run_id)
        report.stats['stage2_cycle_summary_stage_repaired'] = repaired
        report.stats['stage2_cycle_summary_verified_count'] = verified_count

        # --- stage1_flag_marker age-based GC (task 1944) ---
        # The stage1_flag_marker pool is written by Stage 1, not by this stage's
        # agent write, so post-write placement is correct (unlike the pre-write
        # summary pool trim above). Runs unconditionally on both full and
        # remediation paths so the pool is bounded every cycle. Explicit zero
        # so downstream consumers never need a .get(..., 0) fallback.
        gc_swept = await _sweep_stale_flag_markers(self.memory, self.project_id, run_id)
        report.stats['stale_flag_markers_gc_swept'] = gc_swept

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
        available), Guards 4 and 4b still fire (pure stats arithmetic).

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
                        logger.warning(
                            'reconciliation._apply_post_flight_guards: '
                            'get_task returned a non-dict for task_id=%s'
                            ' (type=%s); guards will skip ops on this task',
                            tid, type(result).__name__,
                        )
                        continue  # non-dict result; omit entry so helpers skip
                    extracted = _extract_status(result)
                    if extracted == 'unknown':
                        logger.warning(
                            'reconciliation._apply_post_flight_guards: '
                            'could not resolve status for task_id=%s'
                            ' (no status key or data.status); guards will skip ops on this task',
                            tid,
                        )
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

        # Guard 4 — analytical flag-counter completeness (pure stats arithmetic, no I/O)
        flag_check = _check_flag_counter_completeness(report.stats, prior_reports)
        if flag_check['mismatch']:
            logger.warning(
                'reconciliation.stage1_analytical_findings_processed_mismatch',
                extra={
                    'run_id': run_id,
                    'project_id': self.project_id,
                    'expected': flag_check['expected'],
                    'reported': flag_check['reported'],
                },
            )
            # Clamp to truth so downstream verifiers see the real picture.
            report.stats['stage1_analytical_findings_processed'] = flag_check['expected']

        # Guard 4b — Mem0 flag-counter completeness (pure stats arithmetic, no I/O)
        # Counts flag_deleted action records in stats['flag_deleted_records'] as the
        # ground-truth source, giving stage1_mem0_flags_processed the same clamp/warn
        # coverage as the analytical counter.
        mem0_check = _check_mem0_flag_counter_completeness(report.stats)
        if mem0_check['mismatch']:
            logger.warning(
                'reconciliation.stage1_mem0_flags_processed_mismatch',
                extra={
                    'run_id': run_id,
                    'project_id': self.project_id,
                    'expected': mem0_check['expected'],
                    'reported': mem0_check['reported'],
                },
            )
            # Clamp to truth so downstream verifiers see the real picture.
            report.stats['stage1_mem0_flags_processed'] = mem0_check['expected']

        # Normalize: ensure both Stage 2 counters are always present in stats so
        # Stage 3's audit sees a deterministic, present pair via recon_report's
        # free-form stats passthrough.  setdefault preserves any agent-reported
        # value (including a clamped value written above by Guard 4 or 4b).
        report.stats.setdefault('stage1_analytical_findings_processed', 0)
        report.stats.setdefault('stage1_mem0_flags_processed', 0)

        # Guard 5 — live-workflow status-write reclassification (task 1655)
        # Detects set_task_status ops where the target task has a live workflow
        # (registered worktree / recent branch commits / orchestrator live).
        # The LLM's write already landed; this guard post-hoc flags the churn
        # (stats + log) for observability.  Actual prevention is the Stage 2 prompt.
        if self.project_root:
            live_workflow_violations = await _classify_live_workflow_status_writes(
                ops, self.project_root, _stage_agent_id
            )
            if live_workflow_violations:
                report.stats['live_workflow_status_writes'] = report.stats.get(
                    'live_workflow_status_writes', 0
                ) + len(live_workflow_violations)
                report.stats['tasks_modified'] = max(
                    0,
                    report.stats.get('tasks_modified', 0) - len(live_workflow_violations),
                )
                for v in live_workflow_violations:
                    logger.info(
                        'reconciliation.live_workflow_status_write_suppressed',
                        extra={
                            'run_id': run_id,
                            'project_id': self.project_id,
                            'task_id': v['task_id'],
                            'op_id': v['op_id'],
                        },
                    )

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

        # Live-Workflow Signals section: check active tasks for live workflows so the
        # Stage 2 LLM can skip set_task_status / stranded-work escalation for those tasks.
        # Only active tasks are inspected (done/cancelled tasks cannot have live workflows).
        # Empty string when no active tasks are live (keeps the payload tight).
        live_workflow_section = ''
        if self.project_root and filtered.active_tasks:
            live_workflow_section = _render_live_workflow_section(
                filtered.active_tasks,
                self.project_root,
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
        # cycle markers are rescued to .current by the run-window guard) and are swept
        # below so they are never rendered to the LLM.
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
        partition = await _query_stage2_flags(
            self.memory, self.project_id, run_id_for_markers,
            run_window_start=run_window_start,
        )
        active_flags = partition.current
        stale_marker_ids = partition.stale_all_ids
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

        # Task 1572: generate a fresh CSPRNG nonce each cycle to supply structural
        # entropy that the monotonic ISO timestamp (tasks 1473/1488) failed to provide.
        # The nonce is generated Python-side (not by the LLM) and injected here so
        # the Stage 2 agent can prepend it as the FIRST line of its per-cycle summary.
        # Prepending shifts the leading content, defeating Mem0 cosine-similarity dedup
        # (~0.92 threshold) that silently dropped structurally-uniform summaries in 8+
        # confirmed recurrences.  The nonce is additive: run_id, flag_id UUIDs, task IDs,
        # and uniqueness_token all remain in the prompt guidance.
        # Task 1574: delegated to build_summary_nonce_section() so Stage 1 and Stage 2
        # share identical section wording and cannot silently drift.
        # Task 1590: passes 'STAGE2' prefix so Stage 2 nonces always lead with 'STAGE2_',
        # which can never collide with Stage 1's 'STAGE1_'-prefixed nonces — making stage
        # origin explicit and further separating the two stages' summaries in Mem0's embedding
        # space.
        summary_nonce_section = build_summary_nonce_section('STAGE2')

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
{active_tree_text}

### Recently Completed Tasks
{recently_completed_text}
{provenance_section}{proactive_sample_section}{hint_conversion_section}{live_workflow_section}{summary_nonce_section}

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
    ) -> StageReport:
        """Execute Stage 3 and post-process with task-dump spot-check.

        Calls super().run() first (LLM agent + report extraction), then:
        1. Applies filter_blocked_snapshot_findings() to suppress false-positive
           task-count snapshot findings for projects with blocked-by-design write
           paths (e.g. autopilot_video).  Records
           report.stats['blocked_snapshot_findings_dropped'] = before - after.
        2. Calls record_task_dump_spot_check() to record a non-destructive
           observability stat when the cached task tree contains contamination
           signals.

        Mirrors MemoryConsolidator.run() override structure (Stage 1).
        """
        report = await super().run(events, watermark, prior_reports, run_id, model=model)

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
