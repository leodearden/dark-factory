"""Past Stage-1 passes, shaped as ``ReconciliationJournal.get_recent_runs`` returns them.

The one test copy of what a persisted preservation-specimen observation looks
like (task 5464): a REAL ``ReconciliationRun`` whose ``memory_consolidator``
``StageReport`` carries the two observation stats the guard reads back.  The
guard's persistence tests and the consolidator's wiring tests both build their
history here, so a change to the observation's shape is made once.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from fused_memory.models.reconciliation import (
    ReconciliationRun,
    RunStatus,
    RunType,
    StageId,
    StageReport,
)
from fused_memory.reconciliation.preservation_specimen_guard import (
    CANDIDATE_TASK_IDS_STAT,
    UNRESOLVED_TASK_IDS_STAT,
)

STAGE1_KEY = StageId.memory_consolidator.value


def observation_report(
    *,
    at: datetime,
    candidates: Iterable[str] = (),
    unresolved: Iterable[str] = (),
    stats: dict[str, Any] | None = None,
) -> StageReport:
    """A Stage-1 report that observed *candidates* plus *unresolved* and left *unresolved* unresolved.

    *stats* replaces the observation outright, for a pre-change or malformed row.
    """
    unresolved = list(unresolved)
    if stats is None:
        stats = {
            CANDIDATE_TASK_IDS_STAT: sorted(set(candidates) | set(unresolved)),
            UNRESOLVED_TASK_IDS_STAT: unresolved,
        }
    return StageReport(
        stage=StageId.memory_consolidator, started_at=at, completed_at=at, stats=stats,
    )


def stage1_pass(
    run_id: str,
    *,
    project_id: str,
    minutes_ago: float,
    candidates: Iterable[str] = (),
    unresolved: Iterable[str] = (),
    stats: dict[str, Any] | None = None,
    stage_reports: dict[str, StageReport | dict] | None = None,
    run_type: RunType = RunType.full,
) -> ReconciliationRun:
    """A completed past run whose Stage-1 report records one observation.

    *stage_reports* replaces the whole report map, for a run that built no
    Stage-1 report.
    """
    at = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    if stage_reports is None:
        stage_reports = {
            STAGE1_KEY: observation_report(
                at=at, candidates=candidates, unresolved=unresolved, stats=stats,
            ),
        }
    return ReconciliationRun(
        id=run_id,
        project_id=project_id,
        run_type=run_type,
        trigger_reason='test',
        started_at=at,
        completed_at=at,
        stage_reports=stage_reports,
        status=RunStatus.completed,
    )


def unresolved_passes(
    *task_ids: str, project_id: str, count: int, start: int = 1,
) -> list[ReconciliationRun]:
    """*count* past passes, most-recent-first, each leaving *task_ids* unresolved."""
    return [
        stage1_pass(f'prior-{n}', project_id=project_id, unresolved=task_ids, minutes_ago=10 * n)
        for n in range(start, start + count)
    ]
