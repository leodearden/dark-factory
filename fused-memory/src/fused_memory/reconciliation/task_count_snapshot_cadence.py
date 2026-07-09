"""Task-count snapshot write-cadence hardening — task 2278.

Reify's Mem0 ``task_count_snapshot`` observation (metadata.kind=
'task_count_snapshot', category='observations_and_summaries') is written
per-project by reconciliation Stage 2 (task_knowledge_sync) as its FINAL
action each cycle.  The write depends entirely on the Stage-2 LLM
*remembering* the (memory-stored) Snapshot Discipline norm — there is no
structural guarantee.  This module provides the pure, dependency-free
building blocks for two structural guards:

1. A Stage-2 freshness stat (``report.stats['task_count_snapshot_written']``)
   computed deterministically in Python from the run-window timestamp of
   existing ``kind='task_count_snapshot'`` Mem0 records — see
   :func:`extract_snapshot_written` and the ``_verify_task_count_snapshot_written``
   helper in ``stages/task_knowledge_sync.py`` that produces the stat.
2. A harness consecutive-full-cycle-miss escalation — see
   :func:`evaluate_snapshot_cadence` and :func:`build_stale_snapshot_finding`,
   wired into ``harness.py``'s ``_maybe_escalate_stale_task_count_snapshot``.

Task 2325 follow-up hardens the write itself rather than only observing it:
:func:`build_task_count_snapshot_content` is the pure renderer for a
DETERMINISTIC Python ``add_memory`` write performed at the end of Stage 2's
``run()`` (the ``_write_task_count_snapshot`` I/O helper in
``stages/task_knowledge_sync.py``), so the write no longer depends on the
Stage-2 LLM remembering the memory-stored "Snapshot Discipline" norm. Gated
on ``not is_snapshot_write_blocked(project_id)`` (see
``reconciliation/policies``) — projects that have never used the per-project
census (currently dark_factory, solar_challenge_platform) are exempted
rather than given a write nobody consumes.

Structural template: mirrors :mod:`fused_memory.reconciliation.stage1_stall_detector`
(threshold constant + pure compute helper + escalation), but per-project (not
per-task) and journal-backed (not Mem0-marker-backed) — see design_decisions
in plan.json for task 2278.

This module has zero imports from ``stages/`` or ``harness`` — it is pure and
side-effect-free so both can import from it without a dependency cycle.
"""

from __future__ import annotations

TASK_COUNT_SNAPSHOT_KIND: str = 'task_count_snapshot'
"""Mem0 metadata ``kind`` tag identifying a task-count snapshot observation."""

SNAPSHOT_WRITTEN_STAT_KEY: str = 'task_count_snapshot_written'
"""Key under Stage 2's ``report.stats`` recording this cycle's freshness check.

Value is ``1`` when a fresh snapshot was confirmed within the run window,
``0`` when confirmed absent, and the key is omitted entirely when the check
was inconclusive (unknown run window or a transient query failure) — see
:func:`extract_snapshot_written`.
"""

TASK_COUNT_SNAPSHOT_MISS_THRESHOLD: int = 2
"""Number of consecutive full-cycle misses before the harness escalates."""

ESCALATION_CATEGORY: str = 'recon_stale_task_count_snapshot'
"""Escalation category for a sustained task_count_snapshot cadence gap.

Registered in ``harness.py``'s ``_RECON_DEDUP_CONFIG.infra_dedupe_categories``
and the 'info'-severity category tuple in ``_escalate`` — this is low-urgency
process/tooling hardening, not an operator-blocking issue.
"""

TASK_COUNT_SNAPSHOT_CATEGORY: str = 'observations_and_summaries'
"""Mem0 ``category`` for the task_count_snapshot write — task 2325.

Used by the deterministic write helper (``_write_task_count_snapshot`` in
``stages/task_knowledge_sync.py``) so the internal ``memory_service.add_memory``
call lands in the same category the pre-existing (LLM-authored) snapshot
writes already use.
"""


def extract_snapshot_written(stage_report: object) -> bool | None:
    """Read the freshness stat off a Stage-2 report.

    Accepts a real ``StageReport`` (attribute access), a raw dict shape (e.g.
    a journal-reconstructed ``_error`` entry or test double), or ``None``.

    Returns:
        ``True`` when ``stats['task_count_snapshot_written'] == 1``,
        ``False`` when ``== 0``, and ``None`` when the report is ``None``,
        the ``stats`` dict is absent, or the key itself is absent —
        "unknown", never miscounted as a confirmed miss.
    """
    if stage_report is None:
        return None
    if isinstance(stage_report, dict):
        stats = stage_report.get('stats') or {}
    else:
        stats = getattr(stage_report, 'stats', None) or {}
    value = stats.get(SNAPSHOT_WRITTEN_STAT_KEY)
    if value == 1:
        return True
    if value == 0:
        return False
    return None


def compute_snapshot_miss_streak(recent_flags: list[bool | None]) -> int:
    """Count the leading run of consecutive misses in *recent_flags*.

    *recent_flags* is most-recent-first.  Counts consecutive ``False``
    entries from the start, stopping at the first ``True`` (a written cycle
    resets the streak) or ``None`` (unknown — stop, fail-safe: an
    inconclusive cycle must never be counted as either a miss or a reset).
    """
    streak = 0
    for flag in recent_flags:
        if flag is False:
            streak += 1
        else:
            break
    return streak


def evaluate_snapshot_cadence(
    current_written: bool | None,
    prior_flags: list[bool | None],
    *,
    blocked: bool,
    threshold: int = TASK_COUNT_SNAPSHOT_MISS_THRESHOLD,
) -> dict:
    """Decide whether the current cycle's miss should escalate.

    Fail-safe / fail-open rules, checked before the streak is ever computed:

    - *blocked* (project is in ``SNAPSHOT_WRITE_BLOCKED_PROJECTS``): the
      absence of a snapshot is correct-by-design there, so never escalate.
    - *current_written* is not ``False`` (i.e. ``True`` — a fresh snapshot
      was confirmed this cycle — or ``None`` — the check was inconclusive):
      never escalate. Only a CONFIRMED current miss can trigger escalation.

    Otherwise the streak is ``compute_snapshot_miss_streak(prior_flags) + 1``
    (the "+1" is the current confirmed miss), and ``escalate`` is
    ``streak >= threshold``.

    Returns:
        ``{'streak': int, 'escalate': bool}``.
    """
    if blocked or current_written is not False:
        return {'streak': 0, 'escalate': False}
    streak = compute_snapshot_miss_streak(prior_flags) + 1
    return {'streak': streak, 'escalate': streak >= threshold}


def build_stale_snapshot_finding(project_id: str) -> dict:
    """Build the stable per-project finding dict for a stale-snapshot escalation.

    Shape mirrors ``harness._DEAD_OWNER_STORM_FINDING``: only ``category``,
    ``affected_ids``, and ``description`` are set, and none of them vary
    across calls for the same *project_id* — this stable identity is what lets
    ``_escalate``'s content-fingerprint dedup fold every repeat cycle's call
    into a single pending escalation instead of filing one per cycle.
    """
    return {
        'category': ESCALATION_CATEGORY,
        'affected_ids': [f'{TASK_COUNT_SNAPSHOT_KIND}:{project_id}'],
        'description': (
            f'task_count_snapshot has not been written for project {project_id!r} '
            'within the run window for multiple consecutive full reconciliation cycles'
        ),
    }


def build_task_count_snapshot_content(
    project_id: str,
    *,
    total: int,
    done: int,
    cancelled: int,
    active: int,
    other: int,
    highest_task_id: int,
    as_of: str | None = None,
) -> str:
    """Render the human-readable content string for a task_count_snapshot write.

    Pure and dependency-free — no I/O, no imports from ``stages/`` or
    ``harness``. Counts are the caller's responsibility (see
    ``_write_task_count_snapshot`` in ``stages/task_knowledge_sync.py``,
    which derives them from ``filter_task_tree``'s
    ``FilteredTaskTree.{total_count,done_count,cancelled_count,other_count,
    max_task_id}`` plus ``len(active_tasks)``).

    Args:
        project_id: Project the census is scoped to.
        total: Total task count.
        done: Count of done tasks.
        cancelled: Count of cancelled tasks.
        active: Count of active (non-terminal) tasks.
        other: Count of tasks in an unrecognized/other status.
        highest_task_id: Highest top-level task id observed.
        as_of: ISO date string for the census, or ``None`` when unknown — the
            leading "As of {as_of}:" clause is omitted in that case.

    Returns:
        A concise, non-empty human-readable census line.
    """
    prefix = f'As of {as_of}: ' if as_of else ''
    return (
        f'{prefix}project {project_id} task-count snapshot — '
        f'{total} total, {done} done, {cancelled} cancelled, {active} active, '
        f'{other} other, highest task id {highest_task_id}.'
    )
