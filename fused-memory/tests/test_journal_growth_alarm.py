"""Tests for the write-journal growth alarm (task 3311).

Measurement is driven through ``WriteJournal.growth_sample`` against a real
journal file; rows with controlled ``created_at`` values are seeded through a
second, stdlib ``sqlite3`` connection on ``journal.db_path``.
"""

import dataclasses
import logging
import os
import sqlite3
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from fused_memory.config.schema import WriteJournalGrowthAlarmConfig
from fused_memory.middleware import _folded_escalation
from fused_memory.services.journal_growth_alarm import (
    JournalCeiling,
    JournalGrowthAlarm,
    JournalGrowthBreach,
    find_breaches,
)
from fused_memory.services.write_journal import JournalGrowthSample, WriteJournal


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = WriteJournal(tmp_path / 'wj')
    await j.initialize()
    yield j
    await j.close()


def seed_ops(db_path: Path, created_ats: Sequence[datetime], *, payload: str = '') -> None:
    """INSERT one read row per timestamp, in the given (strictly increasing) order."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO write_ops (id, operation, kind, params, created_at) "
            "VALUES (?, 'get_task', 'read', ?, ?)",
            [(str(uuid.uuid4()), payload, ts.isoformat()) for ts in created_ats],
        )
        conn.commit()
    finally:
        conn.close()


def _size_or_zero(path: Path) -> int:
    try:
        return os.stat(path).st_size
    except FileNotFoundError:
        return 0


def _on_disk_bytes(db_path: Path) -> int:
    return sum(
        _size_or_zero(db_path.with_name(db_path.name + suffix))
        for suffix in ('', '-wal', '-shm')
    )


def _free_bytes(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        freelist_count = conn.execute('PRAGMA freelist_count').fetchone()[0]
        page_size = conn.execute('PRAGMA page_size').fetchone()[0]
    finally:
        conn.close()
    return freelist_count * page_size


def _an_hour_ago() -> datetime:
    return datetime.now(UTC) - timedelta(hours=1)


class TestGrowthSampleBytes:
    @pytest.mark.asyncio
    async def test_db_path_names_the_journal_file(self, journal):
        assert journal.db_path == journal.data_dir / 'write_journal.db'
        assert journal.db_path.is_file()

    @pytest.mark.asyncio
    async def test_file_bytes_sums_the_db_and_its_sidecars(self, journal):
        seed_ops(journal.db_path, [_an_hour_ago()], payload='x' * 1024)

        sample = await journal.growth_sample(since=_an_hour_ago())

        assert sample.file_bytes == _on_disk_bytes(journal.db_path)
        assert sample.file_bytes >= _size_or_zero(journal.db_path) > 0

    @pytest.mark.asyncio
    async def test_free_bytes_is_the_freelist_in_bytes(self, journal):
        base = _an_hour_ago()
        seed_ops(
            journal.db_path,
            [base + timedelta(milliseconds=i) for i in range(2000)],
            payload='p' * 1024,
        )
        conn = sqlite3.connect(journal.db_path)
        try:
            conn.execute('DELETE FROM write_ops')
            conn.commit()
        finally:
            conn.close()

        sample = await journal.growth_sample(since=base)

        assert sample.free_bytes > 0
        assert sample.free_bytes == _free_bytes(journal.db_path)

    def test_a_sample_is_frozen(self):
        sample = JournalGrowthSample(
            file_bytes=1, free_bytes=0, rows_inserted=0, since=datetime.now(UTC)
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample.file_bytes = 2  # type: ignore[misc]


_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _seconds_after_t0(*offsets: float) -> list[datetime]:
    return [_T0 + timedelta(seconds=s) for s in offsets]


def _delete_ops_at(db_path: Path, created_ats: Sequence[datetime]) -> None:
    """Delete rows the way a retention prune would, through a second connection."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            'DELETE FROM write_ops WHERE created_at = ?',
            [(ts.isoformat(),) for ts in created_ats],
        )
        conn.commit()
    finally:
        conn.close()


class TestGrowthSampleRowsInserted:
    @pytest.mark.asyncio
    async def test_counts_the_rows_inserted_at_or_after_since(self, journal):
        seed_ops(journal.db_path, _seconds_after_t0(1, 2, 3, 4, 5, 6))

        sample = await journal.growth_sample(since=_T0 + timedelta(seconds=3.5))

        assert sample.rows_inserted == 3

    @pytest.mark.asyncio
    async def test_a_cutoff_after_the_newest_row_counts_nothing(self, journal):
        seed_ops(journal.db_path, _seconds_after_t0(1, 2, 3))

        sample = await journal.growth_sample(since=_T0 + timedelta(seconds=10))

        assert sample.rows_inserted == 0

    @pytest.mark.asyncio
    async def test_an_empty_journal_counts_nothing(self, journal):
        sample = await journal.growth_sample(since=_T0)

        assert sample.rows_inserted == 0

    @pytest.mark.asyncio
    async def test_counts_inserts_not_live_rows(self, journal):
        seed_ops(journal.db_path, _seconds_after_t0(1, 2, 3, 4, 5, 6))
        since = _T0 + timedelta(seconds=3.5)

        _delete_ops_at(journal.db_path, _seconds_after_t0(1, 2))
        assert (await journal.growth_sample(since=since)).rows_inserted == 3

        _delete_ops_at(journal.db_path, _seconds_after_t0(5))
        assert (await journal.growth_sample(since=since)).rows_inserted == 3

    @pytest.mark.asyncio
    async def test_counts_rows_logged_through_the_journal(self, journal):
        seed_ops(journal.db_path, _seconds_after_t0(1, 2, 3))
        for operation in ('get_task', 'get_statuses'):
            await journal.log_write_op(
                write_op_id=str(uuid.uuid4()), operation=operation, kind='read'
            )

        sample = await journal.growth_sample(since=_an_hour_ago())

        assert sample.rows_inserted == 2


def _sample(*, file_bytes: int = 100, rows_inserted: int = 10) -> JournalGrowthSample:
    return JournalGrowthSample(
        file_bytes=file_bytes, free_bytes=7, rows_inserted=rows_inserted, since=_T0
    )


_CEILINGS = WriteJournalGrowthAlarmConfig(max_file_bytes=100, max_rows_inserted_per_day=10)


class TestFindBreaches:
    def test_at_or_below_both_ceilings_is_no_breach(self):
        assert find_breaches(_sample(file_bytes=100, rows_inserted=10), _CEILINGS) == ()
        assert find_breaches(_sample(file_bytes=1, rows_inserted=0), _CEILINGS) == ()

    def test_a_file_over_its_ceiling_is_a_size_breach(self):
        sample = _sample(file_bytes=101)

        assert find_breaches(sample, _CEILINGS) == (
            JournalGrowthBreach(ceiling=JournalCeiling.FILE_SIZE, measured=101, limit=100),
        )

    def test_inserts_over_their_ceiling_are_a_rate_breach(self):
        sample = _sample(rows_inserted=11)

        assert find_breaches(sample, _CEILINGS) == (
            JournalGrowthBreach(ceiling=JournalCeiling.INSERT_RATE, measured=11, limit=10),
        )

    def test_both_over_reports_size_then_rate(self):
        breaches = find_breaches(_sample(file_bytes=101, rows_inserted=11), _CEILINGS)

        assert [b.ceiling for b in breaches] == [
            JournalCeiling.FILE_SIZE,
            JournalCeiling.INSERT_RATE,
        ]

    def test_a_breach_is_frozen(self):
        breach = JournalGrowthBreach(ceiling=JournalCeiling.FILE_SIZE, measured=2, limit=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            breach.measured = 3  # type: ignore[misc]


_ALARM_LOGGER = 'fused_memory.services.journal_growth_alarm'
_NEVER = 10**15


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == _ALARM_LOGGER and r.levelno == logging.WARNING
    ]


def _size_words(sample: JournalGrowthSample) -> list[str]:
    return [
        f'{sample.file_bytes:,} B',
        f'{sample.file_bytes / 2**30:.2f} GiB',
        f'{sample.free_bytes:,} B free',
    ]


async def _log_reads(journal: WriteJournal, count: int) -> None:
    for _ in range(count):
        await journal.log_write_op(
            write_op_id=str(uuid.uuid4()), operation='get_task', kind='read'
        )


class TestCheck:
    @pytest.mark.asyncio
    async def test_under_both_ceilings_is_silent(self, journal, caplog):
        caplog.set_level(logging.WARNING, logger=_ALARM_LOGGER)
        config = WriteJournalGrowthAlarmConfig(
            max_file_bytes=_NEVER, max_rows_inserted_per_day=_NEVER
        )

        assert await JournalGrowthAlarm(journal, config, project_root=None).check() == ()
        assert _warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_a_size_breach_warns_naming_the_measured_size(self, journal, caplog):
        caplog.set_level(logging.WARNING, logger=_ALARM_LOGGER)
        config = WriteJournalGrowthAlarmConfig(
            max_file_bytes=4097, max_rows_inserted_per_day=_NEVER
        )

        breaches = await JournalGrowthAlarm(journal, config, project_root=None).check()
        sample = await journal.growth_sample(since=_an_hour_ago())

        assert [b.ceiling for b in breaches] == [JournalCeiling.FILE_SIZE]
        assert breaches[0].measured == sample.file_bytes
        [message] = _warnings(caplog)
        for words in [*_size_words(sample), 'ceiling of 4,097 B']:
            assert words in message

    @pytest.mark.asyncio
    async def test_a_rate_breach_warns_naming_the_count_ceiling_and_size(
        self, journal, caplog
    ):
        caplog.set_level(logging.WARNING, logger=_ALARM_LOGGER)
        await _log_reads(journal, 12)
        config = WriteJournalGrowthAlarmConfig(
            max_file_bytes=_NEVER, max_rows_inserted_per_day=11
        )

        breaches = await JournalGrowthAlarm(journal, config, project_root=None).check()
        sample = await journal.growth_sample(since=_an_hour_ago())

        assert breaches == (
            JournalGrowthBreach(ceiling=JournalCeiling.INSERT_RATE, measured=12, limit=11),
        )
        [message] = _warnings(caplog)
        for words in [*_size_words(sample), '12 inserts in the trailing 24 h', 'ceiling of 11/day']:
            assert words in message

    @pytest.mark.asyncio
    async def test_both_breached_warns_once_per_ceiling(self, journal, caplog):
        caplog.set_level(logging.WARNING, logger=_ALARM_LOGGER)
        await _log_reads(journal, 2)
        config = WriteJournalGrowthAlarmConfig(max_file_bytes=1, max_rows_inserted_per_day=1)

        breaches = await JournalGrowthAlarm(journal, config, project_root=None).check()

        assert [b.ceiling for b in breaches] == [
            JournalCeiling.FILE_SIZE,
            JournalCeiling.INSERT_RATE,
        ]
        assert len(_warnings(caplog)) == 2

    @pytest.mark.asyncio
    async def test_a_failed_sample_is_logged_not_raised(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger=_ALARM_LOGGER)
        alarm = JournalGrowthAlarm(
            WriteJournal(tmp_path / 'never_initialized'),
            WriteJournalGrowthAlarmConfig(max_file_bytes=1),
            project_root=None,
        )

        assert await alarm.check() == ()
        errors = [
            r for r in caplog.records
            if r.name == _ALARM_LOGGER and r.levelno == logging.ERROR
        ]
        assert len(errors) == 1
        assert errors[0].exc_info is not None


_needs_escalation = pytest.mark.skipif(
    not _folded_escalation.HAS_ESCALATION,
    reason='escalation package unavailable (minimal env); the HAS_ESCALATION '
           'no-op arm is covered separately below',
)


def _pending(tmp_path: Path) -> list:
    from escalation.queue import EscalationQueue  # noqa: PLC0415

    return EscalationQueue(tmp_path / 'data' / 'escalations').get_pending()


def _detail_fields(detail: str) -> dict[str, str]:
    return dict(
        line.split('=', 1) for line in detail.splitlines() if '=' in line.split(' ', 1)[0]
    )


@_needs_escalation
class TestEscalation:
    @pytest.mark.asyncio
    async def test_a_size_breach_files_one_operator_escalation(self, journal, tmp_path):
        config = WriteJournalGrowthAlarmConfig(
            max_file_bytes=4097, max_rows_inserted_per_day=_NEVER
        )

        [breach] = await JournalGrowthAlarm(
            journal, config, project_root=str(tmp_path)
        ).check()
        sample = await journal.growth_sample(since=_an_hour_ago())

        [esc] = _pending(tmp_path)
        assert esc.agent_role == 'fused-memory/write-journal-growth-alarm'
        assert esc.severity == 'blocking'
        assert esc.category == 'risk_identified'
        assert esc.level == 1
        assert f'{breach.measured:,} B' in esc.summary
        assert esc.suggested_action
        fields = _detail_fields(esc.detail)
        assert fields['file_bytes'] == str(breach.measured)
        assert fields['free_bytes'] == str(sample.free_bytes)
        assert fields['ceiling'] == 'file_size'
        assert fields['limit'] == '4097'
        assert fields['db_path'] == str(journal.db_path)
        trailing_day = datetime.now(UTC) - timedelta(days=1)
        assert abs(datetime.fromisoformat(fields['since']) - trailing_day) < timedelta(minutes=5)
        for words in (
            'write_journal_growth_alarm',
            'fused-memory/config/config.yaml',
            'deleted nothing',
            'VACUUM',
        ):
            assert words in esc.detail

    @pytest.mark.asyncio
    async def test_a_persisting_breach_folds_into_the_open_record(self, journal, tmp_path):
        alarm = JournalGrowthAlarm(
            journal,
            WriteJournalGrowthAlarmConfig(max_file_bytes=1, max_rows_inserted_per_day=_NEVER),
            project_root=str(tmp_path),
        )

        await alarm.check()
        await alarm.check()

        assert len(_pending(tmp_path)) == 1

    @pytest.mark.asyncio
    async def test_each_ceiling_keeps_its_own_record(self, journal, tmp_path):
        await _log_reads(journal, 2)
        alarm = JournalGrowthAlarm(
            journal,
            WriteJournalGrowthAlarmConfig(max_file_bytes=1, max_rows_inserted_per_day=1),
            project_root=str(tmp_path),
        )

        await alarm.check()
        await alarm.check()

        pending = _pending(tmp_path)
        assert sorted(_detail_fields(e.detail)['ceiling'] for e in pending) == [
            'file_size',
            'insert_rate',
        ]
        assert len({e.task_id for e in pending}) == 2


class TestEscalationFailSoft:
    @pytest.mark.asyncio
    async def test_without_the_escalation_package_the_warning_still_fires(
        self, journal, tmp_path, caplog, monkeypatch
    ):
        monkeypatch.setattr(_folded_escalation, 'HAS_ESCALATION', False)
        caplog.set_level(logging.WARNING, logger=_ALARM_LOGGER)
        alarm = JournalGrowthAlarm(
            journal,
            WriteJournalGrowthAlarmConfig(max_file_bytes=1, max_rows_inserted_per_day=_NEVER),
            project_root=str(tmp_path),
        )

        breaches = await alarm.check()

        assert [b.ceiling for b in breaches] == [JournalCeiling.FILE_SIZE]
        assert len(_warnings(caplog)) == 1
