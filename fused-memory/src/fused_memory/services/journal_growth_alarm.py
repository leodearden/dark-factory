"""Make write-journal growth loud: compare a growth sample against configured ceilings.

ALARM ONLY. Nothing here deletes a row or rewrites the file. The retention
prune is ``services/write_journal.py::WriteJournal.prune_write_ops``, and the
volume fix and reclaim are task 5405's. The measured basis of both ceilings
lives in the ``write_journal_growth_alarm`` block of
``fused-memory/config/config.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fused_memory.config.schema import WriteJournalGrowthAlarmConfig
from fused_memory.services.write_journal import JournalGrowthSample


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
