"""Where a completed full cycle starts the NEXT cycle's episode window, and why.

The next cycle's Stage 1 surfaces the episodes whose ``created_at`` is later
than ``watermark.last_episode_timestamp``. An episode that was unreadable at
this cycle's Stage 1 fetch and carries a ``created_at`` at or before that
cursor is never shown to any cycle: a silent, permanent drop.

Task 4465 anchored the cursor at ``run.started_at`` instead of at completion,
which covers every episode CREATED after the fetch. It does not cover ingestion
lag. graphiti_core's ``add_episode`` stamps ``created_at`` when the call
BEGINS, but persists the node only at the END, after LLM extraction. A write
that began before ``run.started_at`` and was still executing at the fetch is
unreadable there, yet dated before the cursor.

So Stage 1 samples, immediately BEFORE its episode fetch, when the earliest
Graphiti write still executing for the project began
(``MemoryService.earliest_unfinished_graphiti_write_start``), and the cursor is
``min(run.started_at, that start)``.

- Sound. An episode unreadable at the fetch either began executing after the
  sample, so its ``created_at`` is later than the sample, which is later than
  ``run.started_at``; or it was executing at the sample, so its ``created_at``
  is no earlier than the recorded start.
- Live. With nothing executing the cursor is exactly ``run.started_at``. A
  write executes for at most the queue's ``write_timeout_seconds``, so the
  cursor lags only by in-flight writes and never freezes on an empty cycle.

The sample travels in Stage 1's report stats, which persist with the run, so a
resumed run (task sigma) whose Stage 1 had already finished anchors on what
that fetch saw.

``last_memory_timestamp`` is deliberately NOT lowered: mem0 stamps a memory's
``created_at`` immediately before inserting it, so a memory is readable as soon
as it is dated.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fused_memory.models.reconciliation import ReconciliationRun, StageId, StageReport

UNFINISHED_WRITE_START_STAT = 'stage1_unfinished_graphiti_write_start'


def unfinished_write_start_stat(start: datetime | None) -> str | None:
    """Stage 1's stat value for *start*: ISO-8601 with offset, or None."""
    return None if start is None else start.astimezone(UTC).isoformat()


def recorded_unfinished_write_start(run: ReconciliationRun) -> datetime | None:
    """The unfinished-write start *run*'s Stage 1 recorded, or None if it recorded none."""
    report = run.stage_reports.get(StageId.memory_consolidator.value)
    if not isinstance(report, StageReport):
        return None
    encoded = report.stats.get(UNFINISHED_WRITE_START_STAT)
    if not isinstance(encoded, str):
        return None
    return datetime.fromisoformat(encoded)


def next_episode_window_anchor(run: ReconciliationRun) -> datetime:
    """Where the cycle after *run* starts its episode window."""
    recorded = recorded_unfinished_write_start(run)
    return run.started_at if recorded is None else min(run.started_at, recorded)
