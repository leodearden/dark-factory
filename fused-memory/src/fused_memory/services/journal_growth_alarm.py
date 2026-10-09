"""Make write-journal growth loud: compare a growth sample against configured ceilings.

ALARM ONLY. Nothing here deletes a row or rewrites the file. The retention
prune is ``services/write_journal.py::WriteJournal.prune_write_ops``, and the
volume fix and reclaim are task 5405's. The measured basis of both ceilings
lives in the ``write_journal_growth_alarm`` block of
``fused-memory/config/config.yaml``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

from fused_memory.config.schema import WriteJournalGrowthAlarmConfig
from fused_memory.services.write_journal import JournalGrowthSample, WriteJournal

logger = logging.getLogger(__name__)

#: The insert-rate ceiling is per day, so the sample window is the trailing day.
_INSERT_WINDOW = timedelta(days=1)
_GIB = 2**30


class JournalCeiling(Enum):
    FILE_SIZE = 'file_size'
    INSERT_RATE = 'insert_rate'


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
        return breaches
