"""Stage 1 stale human-operator-required detector — task 1201.

Provides helpers that detect when a task has required human operator action
for N consecutive Stage 1 cycles and submits a level-1 escalation so the
alert reaches a human rather than silently repeating.

Design decisions (captured in plan.json):
- Uses a separate ``stage1_human_operator_stall_marker`` Mem0 source that
  accumulates monotonically, mirroring Stage 2's ``stage2_persistence_marker``
  pattern (``_track_flag_persistence`` in task_knowledge_sync.py).
- Filters on ``resolution_status == 'human_operator_required'`` (NOT
  ``flag_type``), matching the convention in
  ``_suppress_same_run_human_operator_dups`` (task_knowledge_sync.py:116-176).
- Per-task_id granularity with ``EscalationQueue.has_open_l1(task_id)`` dedup
  so each stalled task receives at most one pending level-1 escalation.
- Graceful import of the optional ``escalation`` package via
  ``HAS_ESCALATION`` / ``try/except ImportError`` per harness.py:47-55.

Escalation category note:
  ``maybe_escalate_stalled_tasks`` submits escalations with
  ``category='reconciliation_stale_human_operator'``.  This is a registered
  member of the canonical category set documented in
  ``escalation/src/escalation/models.py`` (canonical list: scope_violation,
  design_concern, cleanup_needed, dependency_discovered, risk_identified,
  infra_issue, reconciliation_stale_human_operator).
  ``Escalation.category`` is a free-form ``str`` field so it will not fail
  validation; the new value is kept distinct from the Stage 2
  ``reconciliation_stale_flag`` category to allow filtering on each
  independently.

  A second, independent Stage 1 category ``reconciliation_stale_gate_backlog``
  (task 3017) is filed by ``maybe_escalate_stalled_gate_backlog`` when a
  blocked human-decision gate task's ``metadata.gate_escalated_at`` stamp has
  aged past ``STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS`` (48h).  It is a
  deterministic age check — no Mem0 marker accumulation, no LLM finding — and
  is deduped per-task via ``has_open_l1(task_id, category=...)``.  Because that
  lookup is category-scoped, an open gate-backlog L1 is never *suppressed by* an
  unrelated open L1 — a ``reconciliation_stale_human_operator`` L1 or the
  DeterministicRunner's born-at-L2 ``milestone_gate`` escalation (distinct
  level/category).  The independence is only guaranteed in that one direction:
  the HOR path (``maybe_escalate_stalled_tasks``) dedups with an
  UN-categorized ``has_open_l1(task_id)`` (its own task-1201 "at most one open
  L1 per task" contract), so a pre-existing open gate-backlog L1 WOULD *suppress*
  a same-task HOR escalation.  In practice the two populations (LLM-finding
  ``human_operator_required`` flags vs. blocked born-at-L2 gates) are disjoint,
  so this asymmetry does not bite.

Stall marker accumulation:
  ``stage1_human_operator_stall_marker`` memories accumulate indefinitely —
  there is no cleanup hook tied to task resolution or escalation resolution.
  This means that if a task transitions out of ``human_operator_required``,
  gets resolved, and later flips back (e.g. via a remediation re-flag), the
  prior cycle count carries over and the next cycle may immediately exceed
  the threshold.  This is **accepted behavior**: ``EscalationQueue.has_open_l1``
  prevents a duplicate level-1 escalation if one is already pending, and
  the cost of a spurious re-escalation for a genuinely-re-stalled task is
  considered lower than the complexity of a cleanup hook.  If per-episode
  isolation is needed in future, scope the count to a time window via
  additional metadata filters on ``count_memories_by_metadata``.

Dedup interaction:
  ``dedup_flags(...)`` runs in ``stages/memory_consolidator.py`` (lines 71-76)
  **before** ``extract_human_operator_task_ids`` is called at line 84.  A
  ``human_operator_required`` flag suppressed by dedup that cycle is therefore
  invisible to ``extract_human_operator_task_ids``, so
  ``track_human_operator_stalls`` does NOT bump the stall counter for that
  cycle.  The ``STAGE1_HUMAN_OPERATOR_STALL_THRESHOLD = 5`` consequently counts
  *"cycles where the flag survived dedup"*, not *"consecutive real stall
  cycles"*.  A genuinely-stalled task can therefore take materially more than 5
  cycles to escalate if dedup intermittently suppresses its flag.  This is
  intentional / accepted behaviour (option (a) from the original review) — no
  behaviour change in this task.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fused_memory.utils.async_utils import gather_collect

try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    _HAS_ESCALATION = True
except ImportError:
    Escalation = None  # type: ignore[assignment,misc]
    _HAS_ESCALATION = False

logger = logging.getLogger(__name__)

# ── Public constants ─────────────────────────────────────────────────────────

STAGE1_HUMAN_OPERATOR_STALL_THRESHOLD: int = 5
"""Number of consecutive Stage 1 cycles with ``human_operator_required`` before
a level-1 escalation is submitted."""

_STAGE1_HUMAN_OPERATOR_STALL_MARKER_SOURCE = 'stage1_human_operator_stall_marker'
"""Mem0 metadata ``source`` tag used to accumulate per-task stall-cycle
markers.  Kept private to this module — callers use the public helpers."""

STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS: float = 48 * 3600
"""Age (in seconds) a blocked human-decision gate task's
``metadata.gate_escalated_at`` stamp may reach before a level-1
``reconciliation_stale_gate_backlog`` escalation is filed (task 3017).  A
module constant, mirroring ``STAGE1_HUMAN_OPERATOR_STALL_THRESHOLD`` (not
config)."""

_GATE_BACKLOG_ESCALATION_CATEGORY = 'reconciliation_stale_gate_backlog'
"""``Escalation.category`` used for the deterministic gate-backlog age check.
Kept distinct from ``reconciliation_stale_human_operator`` so the two Stage 1
aging signals dedup and filter independently (task 3017)."""


# ── Pure helpers ─────────────────────────────────────────────────────────────


def extract_human_operator_task_ids(flags: list[dict]) -> list[str]:
    """Return distinct str task_ids from flags whose resolution_status is HOR.

    Filtering convention: ``resolution_status == 'human_operator_required'``
    (see ``_suppress_same_run_human_operator_dups`` in task_knowledge_sync.py).

    Rules:
    - Skip flags whose ``resolution_status`` is not ``'human_operator_required'``.
    - Skip flags with missing or None ``task_id``.
    - Coerce int task_ids to str (so int 0 and str '0' collapse).
    - Deduplicate: first-seen order preserved; subsequent occurrences of the
      same str task_id are silently dropped.
    - Empty input returns [].
    """
    seen: set[str] = set()
    result: list[str] = []
    for flag in flags:
        if flag.get('resolution_status') != 'human_operator_required':
            continue
        raw_tid = flag.get('task_id')
        if raw_tid is None:
            continue
        tid = str(raw_tid)
        if not tid:
            continue
        if tid not in seen:
            seen.add(tid)
            result.append(tid)
    return result


def gate_escalated_age_secs(metadata: dict | None, *, now: datetime) -> float | None:
    """Age (in seconds) of ``metadata['gate_escalated_at']``, or ``None``.

    ``gate_escalated_at`` is a DeterministicRunner idempotency stamp written
    ONLY by ``_file_milestone_gate_and_block`` whenever a born-at-L2
    human-decision gate is filed (blessed key
    ``shared/src/shared/task_metadata.py``).  Its presence is therefore the
    unambiguous, authoritative "a human decision has been pending since this
    instant" signal (task 3017), so the age check keys purely on a parseable
    stamp — it deliberately does NOT filter on ``operational_mode``.

    Rationale for ignoring ``operational_mode`` (reviewer amendment): the
    operational-routing contract
    (``fused_memory/middleware/operational_routing_guard.py``) coerces a
    ``decision`` task (ANY mode) and an ``operational`` + ``llm`` task into a
    *pure gate* that ``_file_milestone_gate_and_block`` blocks and stamps —
    while leaving ``operational_mode='llm'`` on the metadata.  The
    LLM-operational lane that would route such a gate to an agent instead of a
    human is explicit future work (``deterministic_runner.py``
    ``operational_llm_needs_lane``), so today an ``operational_mode=='llm'``
    gate STILL awaits a human decision.  An ``operational_mode``-based filter
    would silently exclude exactly that population and defeat the feature; the
    ``gate_escalated_at`` stamp already captures the "human gate filed" fact
    precisely.

    Returns ``None`` — never raises — when:

    - *metadata* is not a ``dict``;
    - ``gate_escalated_at`` is missing, ``None``, empty, or not a ``str``;
    - ``gate_escalated_at`` is not a parseable ISO-8601 timestamp
      (``ValueError`` swallowed);
    - subtracting an aware *now* from a naive parsed stamp raises
      ``TypeError`` (swallowed) — a naive stamp yields ``None``.

    The helper does NOT clamp: a future ``gate_escalated_at`` returns a
    negative age.  Swallowing parse/subtraction errors upholds the module's
    fail-soft contract so one malformed stamp never aborts the Stage 1 pass.
    """
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get('gate_escalated_at')
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
        return (now - parsed).total_seconds()
    except (ValueError, TypeError):
        return None


def extract_stalled_gate_backlog_task_ids(
    tasks: list[dict],
    *,
    now: datetime,
    threshold_secs: float = STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS,
) -> list[str]:
    """Return sorted, deduped str task_ids of blocked gate tasks aged past threshold.

    A task qualifies when ALL hold (task 3017):

    - ``task['status'] == 'blocked'`` (a human-decision gate blocks while its
      born-at-L2 escalation is open);
    - ``gate_escalated_age_secs(task['metadata'], now=now)`` is not ``None``
      (i.e. it carries a parseable ``gate_escalated_at`` stamp — the
      authoritative born-at-L2 human-gate signal, regardless of
      ``operational_mode``) AND that age is strictly greater than
      *threshold_secs*;
    - ``task['id']`` is not ``None`` (coerced to ``str``).

    Pure (no I/O).  Mirrors ``compute_stalled_task_ids``' sorted return and
    ``extract_human_operator_task_ids``' first-seen dedup + ``str(id)``
    coercion.  Callers restrict *tasks* to ``filtered_task_tree.active_tasks``
    so resolved-and-done gates are excluded for free.  Empty input → ``[]``.
    """
    seen: set[str] = set()
    result: list[str] = []
    for task in tasks:
        if task.get('status') != 'blocked':
            continue
        age = gate_escalated_age_secs(task.get('metadata'), now=now)
        if age is None or age <= threshold_secs:
            continue
        raw_tid = task.get('id')
        if raw_tid is None:
            continue
        tid = str(raw_tid)
        if tid not in seen:
            seen.add(tid)
            result.append(tid)
    return sorted(result)


# ── Async I/O helpers ────────────────────────────────────────────────────────


async def track_human_operator_stalls(
    memory_service,
    project_id: str,
    run_id: str,
    task_ids: list[str],
) -> dict[str, int]:
    """Track how many Stage 1 cycles each task has required human operator action.

    For each *task_id* in *task_ids*:

    1. Counts prior ``stage1_human_operator_stall_marker`` memories via a
       deterministic Qdrant payload-filter count
       (``MemoryService.count_memories_by_metadata``).  Uses metadata-filtered
       count rather than semantic search to avoid top-N truncation silently
       dropping prior markers (same rationale as ``_track_flag_persistence``).
    2. Writes a fresh marker so the count accumulates across cycles.
    3. Returns ``{task_id: prior_count + 1}`` where "+1" is the current cycle.

    Both phases are parallelised via ``asyncio.gather(..., return_exceptions=True)``
    so a single Qdrant failure degrades gracefully.

    On count failure: prior_count defaults to 0 (cycle count becomes 1).
    On add_memory failure: the current-cycle count is still returned, but no
    marker is persisted — so the next cycle will again observe prior_count=0
    and return 1.  The counter is therefore **best-effort and not guaranteed
    monotonic across cycles during sustained write failures**.  Both failures
    log WARNING; a persistent write failure can be distinguished from a
    genuinely-non-stale task only by correlating the WARNING log stream.

    Empty *task_ids* returns ``{}`` with no I/O.
    """
    if not task_ids:
        return {}

    # ── count phase (parallel) ───────────────────────────────────────────────
    # Two-tier check via gather_collect (fused_memory.utils.async_utils).
    # Pass 1 (inside gather_collect): re-raises structured-cancellation
    # signals — this preserves the structured-cancellation contract and
    # prevents the stall counter from silently converting a shutdown signal
    # into a wrongly-defaulted prior_count of 0.
    # Pass 2 (below): per-item degrade-to-warning on ordinary Exceptions.
    count_results = await gather_collect(
        memory_service.count_memories_by_metadata(
            project_id=project_id,
            filters={
                'source': _STAGE1_HUMAN_OPERATOR_STALL_MARKER_SOURCE,
                'task_id': tid,
            },
        )
        for tid in task_ids
    )
    prior_counts: dict[str, int] = {}
    for tid, result in zip(task_ids, count_results, strict=True):
        if isinstance(result, Exception):
            logger.warning(
                'stage1_stall_detector: count failed for task_id=%s; treating prior_count as 0',
                tid,
                extra={'project_id': project_id, 'task_id': tid},
            )
            prior_counts[tid] = 0
        else:
            prior_counts[tid] = result

    # ── write phase (parallel) ───────────────────────────────────────────────
    # Two-tier check via gather_collect (fused_memory.utils.async_utils).
    # Pass 1 (inside gather_collect): re-raises structured-cancellation
    # signals — this preserves the structured-cancellation contract and
    # prevents the stall marker write from silently converting a shutdown
    # signal into a silently-dropped marker.
    # Pass 2 (below): per-item degrade-to-warning on ordinary Exceptions.
    write_results = await gather_collect(
        memory_service.add_memory(
            content=f'Stage 1 human-operator stall marker: task={tid} run={run_id}',
            category='observations_and_summaries',
            project_id=project_id,
            metadata={
                'source': _STAGE1_HUMAN_OPERATOR_STALL_MARKER_SOURCE,
                'task_id': tid,
                'run_id': run_id,
            },
            causation_id=run_id,
            _source='stage1_stall_detector',
        )
        for tid in task_ids
    )
    for tid, result in zip(task_ids, write_results, strict=True):
        if isinstance(result, Exception):
            logger.warning(
                'stage1_stall_detector: add_memory failed for task_id=%s; count still returned',
                tid,
                extra={'project_id': project_id, 'task_id': tid},
            )

    return {tid: prior_counts[tid] + 1 for tid in task_ids}


def compute_stalled_task_ids(
    stall_counts: dict[str, int],
    threshold: int = STAGE1_HUMAN_OPERATOR_STALL_THRESHOLD,
) -> list[str]:
    """Return sorted task_ids whose stall_count meets or exceeds *threshold*.

    Pure (no I/O).  Mirrors ``_compute_stale_flags`` in task_knowledge_sync.py.
    """
    return sorted(tid for tid, count in stall_counts.items() if count >= threshold)


async def maybe_escalate_stalled_tasks(
    escalation_queue,
    project_id: str,
    run_id: str,
    stalled_task_ids: list[str],
    stall_counts: dict[str, int],
    flags: list[dict],
) -> list[str]:
    """Submit level-1 escalations for stalled tasks that don't already have one.

    For each task_id in *stalled_task_ids*:
    - Skip (log INFO) if ``escalation_queue.has_open_l1(task_id)`` is truthy.
    - Otherwise build an ``Escalation`` with ``level=1``,
      ``severity='blocking'``, ``category='reconciliation_stale_human_operator'``
      and submit via ``escalation_queue.submit(...)``.  Any submit failure
      logs WARNING and excludes the task_id from the returned list.

    Returns the list of task_ids that actually received a new escalation.
    Returns ``[]`` immediately when the ``escalation`` package is unavailable.
    """
    if Escalation is None:
        return []

    # Build first-seen flag lookup for representative descriptions
    flag_by_task: dict[str, dict] = {}
    for flag in flags:
        raw_tid = flag.get('task_id')
        if raw_tid is None:
            continue
        tid = str(raw_tid)
        if tid not in flag_by_task:
            flag_by_task[tid] = flag

    escalated: list[str] = []
    for task_id in stalled_task_ids:
        if escalation_queue.has_open_l1(task_id):
            logger.info(
                'reconciliation.stage1_human_operator_escalation_suppressed: '
                'task_id=%s already has open level-1 escalation',
                task_id,
                extra={'project_id': project_id, 'task_id': task_id},
            )
            continue

        cycle_count = stall_counts.get(task_id, 0)
        rep_flag = flag_by_task.get(task_id, {})
        flag_description = rep_flag.get('description', '')
        flag_type = rep_flag.get('flag_type', 'unknown')

        summary = (
            f'Task {task_id} stalled in human_operator_required for {cycle_count} cycles'
        )
        detail_parts = [
            f'project_id: {project_id}',
            f'run_id: {run_id}',
            f'cycle_count: {cycle_count}',
            f'flag_type: {flag_type}',
        ]
        if flag_description:
            detail_parts.append(f'flag_description: {flag_description}')
        detail = '\n'.join(detail_parts)

        try:
            # make_id() and Escalation() are inside the try so that id-generation
            # or constructor failures are caught and logged rather than escaping.
            esc = Escalation(
                id=escalation_queue.make_id(task_id),
                task_id=task_id,
                agent_role='reconciliation-stage1',
                severity='blocking',
                category='reconciliation_stale_human_operator',
                summary=summary,
                detail=detail,
                level=1,
            )
            escalation_queue.submit(esc)
            escalated.append(task_id)
        except Exception as exc:
            logger.warning(
                'stage1_stall_detector: failed to escalate task_id=%s (id-gen, construction, or submit): %s',
                task_id,
                exc,
                extra={'project_id': project_id, 'task_id': task_id},
            )

    return escalated


async def maybe_escalate_stalled_gate_backlog(
    escalation_queue,
    project_id: str,
    run_id: str,
    stalled_task_ids: list[str],
    task_by_id: dict[str, dict],
    *,
    now: datetime,
    threshold_secs: float = STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS,
) -> list[str]:
    """Submit level-1 gate-backlog escalations for stalled gate tasks lacking one.

    Mirrors ``maybe_escalate_stalled_tasks`` (task 3017).  For each *task_id* in
    *stalled_task_ids*:

    - Skip (log INFO) when
      ``escalation_queue.has_open_l1(task_id, category=_GATE_BACKLOG_ESCALATION_CATEGORY)``
      is truthy.  The category scope (task 2757) guarantees at most one open
      gate-backlog L1 per task and keeps this path from being *suppressed by* an
      unrelated open L1 or the DeterministicRunner's born-at-L2
      ``milestone_gate`` escalation (distinct level/category).  The independence
      is one-directional: the HOR path (``maybe_escalate_stalled_tasks``) dedups
      with an UN-categorized ``has_open_l1(task_id)``, so an open gate-backlog L1
      CAN *suppress* a same-task HOR escalation — harmless because the two
      populations are disjoint in practice.
    - Otherwise build an ``Escalation`` with ``level=1``, ``severity='blocking'``,
      ``category=_GATE_BACKLOG_ESCALATION_CATEGORY`` and submit it.  The summary
      states the ``gate_escalated_at`` anchor from
      ``task_by_id[task_id]['metadata']`` together with the static
      ``threshold_secs`` fact — both true at any later read time, including
      through a ``compact=True`` drain, which keeps ``summary`` but drops
      ``detail`` (see ``_COMPACT_ESCALATION_FIELDS`` in
      ``escalation/src/escalation/server.py``).  Naming the threshold alongside
      the anchor lets a compact-drain read convey urgency without requiring a
      mental diff against "now".  The age recomputed via
      ``gate_escalated_age_secs`` is retained only as a filing-time forensic
      value in the detail block (``age_hours_at_filing``), since it goes stale
      while the escalation sits open.  The ``age_hours is not None`` branch
      condition doubles as a guard: it holds only when ``gate_escalated_at``
      parsed successfully, so the summary anchor is provably a real timestamp;
      an unparseable or absent stamp falls through to the threshold-named
      fallback summary below instead.  Any failure (id-gen, construction, or
      submit) logs WARNING and excludes the task_id from the returned list.

    Returns the list of task_ids that received a new escalation.  Returns
    ``[]`` immediately when the ``escalation`` package is unavailable.

    *threshold_secs* is accepted for signature symmetry with
    ``extract_stalled_gate_backlog_task_ids`` (the caller has already applied
    the age filter) and names the boundary in both summary branches: paired
    with the live anchor when the age is known, alone in the fallback summary
    when it cannot be recomputed.
    """
    if Escalation is None:
        return []

    escalated: list[str] = []
    for task_id in stalled_task_ids:
        if escalation_queue.has_open_l1(
            task_id, category=_GATE_BACKLOG_ESCALATION_CATEGORY
        ):
            logger.info(
                'reconciliation.stage1_gate_backlog_escalation_suppressed: '
                'task_id=%s already has open level-1 gate-backlog escalation',
                task_id,
                extra={'project_id': project_id, 'task_id': task_id},
            )
            continue

        task = task_by_id.get(task_id) or {}
        metadata = task.get('metadata') if isinstance(task, dict) else None
        age_secs = gate_escalated_age_secs(metadata, now=now)
        age_hours = round(age_secs / 3600, 1) if age_secs is not None else None
        gate_escalated_at = (
            metadata.get('gate_escalated_at') if isinstance(metadata, dict) else None
        )

        if age_hours is not None:
            summary = (
                f'Gate task {task_id} has awaited a human decision since '
                f'{gate_escalated_at} (past the {threshold_secs / 3600:.0f}h '
                f'gate-backlog threshold)'
            )
        else:
            summary = (
                f'Gate task {task_id} has awaited a human decision beyond the '
                f'{threshold_secs / 3600:.0f}h gate-backlog threshold'
            )

        detail_parts = [
            f'project_id: {project_id}',
            f'run_id: {run_id}',
            f'task_id: {task_id}',
            f'gate_escalated_at: {gate_escalated_at}',
            f'age_hours_at_filing: {age_hours if age_hours is not None else "unknown"}',
        ]
        title = task.get('title') if isinstance(task, dict) else None
        if title:
            detail_parts.append(f'title: {title}')
        description = task.get('description') if isinstance(task, dict) else None
        if description:
            detail_parts.append(f'description: {description}')
        detail = '\n'.join(detail_parts)

        try:
            # make_id() and Escalation() are inside the try so that id-generation
            # or constructor failures are caught and logged rather than escaping.
            esc = Escalation(
                id=escalation_queue.make_id(task_id),
                task_id=task_id,
                agent_role='reconciliation-stage1',
                severity='blocking',
                category=_GATE_BACKLOG_ESCALATION_CATEGORY,
                summary=summary,
                detail=detail,
                level=1,
            )
            escalation_queue.submit(esc)
            escalated.append(task_id)
        except Exception as exc:
            logger.warning(
                'stage1_stall_detector: failed to escalate gate-backlog task_id=%s '
                '(id-gen, construction, or submit): %s',
                task_id,
                exc,
                extra={'project_id': project_id, 'task_id': task_id},
            )

    return escalated
