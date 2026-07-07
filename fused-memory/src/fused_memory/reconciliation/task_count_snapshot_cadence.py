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
