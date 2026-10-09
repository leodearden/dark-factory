"""Make write-journal growth loud: compare a growth sample against configured ceilings.

ALARM ONLY. Nothing here deletes a row or rewrites the file. The retention
prune is ``services/write_journal.py::WriteJournal.prune_write_ops``, and the
volume fix and reclaim are task 5405's. The measured basis of both ceilings
lives in the ``write_journal_growth_alarm`` block of
``fused-memory/config/config.yaml``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

from fused_memory.config.schema import WriteJournalGrowthAlarmConfig
from fused_memory.middleware._folded_escalation import file_folded_escalation
from fused_memory.services.write_journal import JournalGrowthSample, WriteJournal

logger = logging.getLogger(__name__)

#: The insert-rate ceiling is per day, so the sample window is the trailing day.
_INSERT_WINDOW = timedelta(days=1)
_GIB = 2**30

_FILE_SIZE_ANCHOR_TASK_ID = 'write-journal-size-ceiling'
_INSERT_RATE_ANCHOR_TASK_ID = 'write-journal-insert-rate-ceiling'
_AGENT_ROLE = 'fused-memory/write-journal-growth-alarm'
_CATEGORY = 'risk_identified'
_BASIS = 'the write_journal_growth_alarm block of fused-memory/config/config.yaml'


class JournalCeiling(Enum):
    FILE_SIZE = 'file_size'
    INSERT_RATE = 'insert_rate'


#: One anchor PER CEILING: a size breach stays open until a VACUUM, and under a
#: shared anchor it would fold every later rate breach into silence.
_ANCHOR_TASK_IDS = {
    JournalCeiling.FILE_SIZE: _FILE_SIZE_ANCHOR_TASK_ID,
    JournalCeiling.INSERT_RATE: _INSERT_RATE_ANCHOR_TASK_ID,
}

_SUGGESTED_ACTIONS = {
    JournalCeiling.FILE_SIZE: (
        'Check whether prune_write_ops is keeping up: its startup WARNING reports '
        'the rows/sec it achieved and which horizon is still behind. Then decide '
        'between an offline VACUUM (fused-memory stopped) and a deliberately raised '
        f'max_file_bytes, recording the reason in {_BASIS}.'
    ),
    JournalCeiling.INSERT_RATE: (
        'Find which operation and agent drove the trailing-24 h insert count, e.g. '
        "via the dashboard memory panel's operations and agent breakdowns. Then "
        'decide whether it is a regression to fix at its source or a new baseline '
        f'that needs max_rows_inserted_per_day re-anchored in {_BASIS}.'
    ),
}


@dataclass(frozen=True)
class JournalGrowthBreach:
    ceiling: JournalCeiling
    measured: int
    limit: int


def find_breaches(
    sample: JournalGrowthSample, config: WriteJournalGrowthAlarmConfig
) -> tuple[JournalGrowthBreach, ...]:
    measurements = (
        (JournalCeiling.FILE_SIZE, sample.file_bytes, config.max_file_bytes),
        (JournalCeiling.INSERT_RATE, sample.rows_inserted, config.max_rows_inserted_per_day),
    )
    return tuple(
        JournalGrowthBreach(ceiling=ceiling, measured=measured, limit=limit)
        for ceiling, measured, limit in measurements
        if measured > limit
    )


def describe_breach(breach: JournalGrowthBreach, sample: JournalGrowthSample) -> str:
    """The one-line account of *breach*: the WARNING text and the escalation summary."""
    size = (
        f'{sample.file_bytes:,} B ({sample.file_bytes / _GIB:.2f} GiB, '
        f'{sample.free_bytes:,} B free)'
    )
    if breach.ceiling is JournalCeiling.FILE_SIZE:
        return f'write_journal.db is {size}, over its file-size ceiling of {breach.limit:,} B'
    return (
        f'write_journal.db took {breach.measured:,} inserts in the trailing 24 h, '
        f'over its ceiling of {breach.limit:,}/day; the file is {size}'
    )


def emit_journal_growth_escalation(
    project_root: str | None,
    breach: JournalGrowthBreach,
    sample: JournalGrowthSample,
    *,
    db_path: Path | None = None,
) -> str | None:
    """File *breach* under its ceiling's anchor, folding into an open record. Never raises."""
    detail = '\n'.join((
        f'file_bytes={sample.file_bytes}',
        f'free_bytes={sample.free_bytes}',
        f'rows_inserted={sample.rows_inserted}',
        f'since={sample.since.isoformat()}',
        f'ceiling={breach.ceiling.value}',
        f'limit={breach.limit}',
        f'db_path={db_path}',
        f'basis={_BASIS}',
        'This alarm deleted nothing. The file shrinks only via an offline VACUUM, '
        'and task 5405 owns the read-row rollup and reclaim.',
    ))
    return file_folded_escalation(
        project_root,
        anchor_task_id=_ANCHOR_TASK_IDS[breach.ceiling],
        agent_role=_AGENT_ROLE,
        category=_CATEGORY,
        severity='blocking',
        summary=describe_breach(breach, sample),
        detail=detail,
        suggested_action=_SUGGESTED_ACTIONS[breach.ceiling],
        logger=logger,
        log_label='write_journal_growth_alarm',
    )


class JournalGrowthAlarm:
    def __init__(
        self,
        journal: WriteJournal,
        config: WriteJournalGrowthAlarmConfig,
        *,
        project_root: str | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._journal = journal
        self._config = config
        self._project_root = project_root
        self._clock = clock
        self._last_check_at: float | None = None

    async def maybe_check(self) -> None:
        """Run :meth:`check` at most once per ``check_interval_seconds``.

        The interval is consumed BEFORE the check runs, so a failing check
        costs one ERROR per interval rather than one per caller tick.
        """
        now = self._clock()
        if (
            self._last_check_at is not None
            and now - self._last_check_at < self._config.check_interval_seconds
        ):
            return
        self._last_check_at = now
        await self.check()

    async def check(self) -> tuple[JournalGrowthBreach, ...]:
        """Sample the journal and WARN once per crossed ceiling. Never raises."""
        try:
            sample = await self._journal.growth_sample(
                since=datetime.now(UTC) - _INSERT_WINDOW
            )
        except Exception:
            logger.exception(
                'write_journal growth alarm: sampling failed; no growth check this cycle'
            )
            return ()
        breaches = find_breaches(sample, self._config)
        for breach in breaches:
            logger.warning('write_journal growth alarm: %s', describe_breach(breach, sample))
        for breach in breaches:
            await self._escalate(breach, sample)
        return breaches

    async def _escalate(self, breach: JournalGrowthBreach, sample: JournalGrowthSample) -> None:
        # The queue write fsyncs, so it runs off the event loop.
        try:
            await asyncio.to_thread(
                emit_journal_growth_escalation,
                self._project_root,
                breach,
                sample,
                db_path=self._journal.db_path,
            )
        except Exception:
            logger.exception(
                'write_journal growth alarm: filing the %s escalation failed',
                breach.ceiling.value,
            )
