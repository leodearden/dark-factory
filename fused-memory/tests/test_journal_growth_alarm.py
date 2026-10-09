"""Tests for the write-journal growth alarm (task 3311).

Measurement is driven through ``WriteJournal.growth_sample`` against a real
journal file; rows with controlled ``created_at`` values are seeded through a
second, stdlib ``sqlite3`` connection on ``journal.db_path``.
"""

import dataclasses
import os
import sqlite3
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

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
