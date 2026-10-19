"""Tests for the write_ops retention policy: its horizons and the pass that runs them."""

import asyncio
import contextlib
import logging
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from fused_memory.config.schema import WriteJournalConfig
from fused_memory.services import write_journal, write_ops_retention
from fused_memory.services.write_journal import WRITE_OPS_PRUNE_BATCH_SQL, WriteJournal
from fused_memory.services.write_ops_retention import (
    prune_write_ops_pass,
    run_periodic_write_ops_prune,
)

_KIND_TIME_RANGE_SEEK = (
    'SEARCH write_ops USING INDEX idx_wo_kind_time (kind=? AND created_at>? AND created_at<?)'
)


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = WriteJournal(tmp_path / 'test_wj')
    await j.initialize()
    yield j
    await j.close()


@pytest.mark.parametrize('horizon', list(WRITE_OPS_PRUNE_BATCH_SQL))
@pytest.mark.asyncio
async def test_every_prune_batch_seeks_between_its_resume_point_and_cutoff(journal, horizon):
    with contextlib.closing(sqlite3.connect(journal.db_path)) as conn:
        plan = conn.execute(
            f'EXPLAIN QUERY PLAN {WRITE_OPS_PRUNE_BATCH_SQL[horizon]}',
            ('2026-01-01T00:00:00+00:00', '2026-02-01T00:00:00+00:00', 1),
        ).fetchall()
    details = [row[3] for row in plan]
    assert _KIND_TIME_RANGE_SEEK in details, (
        f'the {horizon} batch plans as {details}. Unless the (kind, created_at) '
        'seek is bounded on both sides, a batch re-walks every row an earlier batch '
        'passed over, which measured 9.4 s cold in one statement on the live journal.'
    )
    assert not any('TEMP B-TREE' in detail for detail in details), (
        f'the {horizon} batch sorts instead of reading created_at order off the index'
    )


async def _seed(
    journal: WriteJournal, *, operation: str, kind: str, age_days: float, success: bool = True
) -> str:
    """One stored row, aged. Since task 5405 a successful non-search read is
    counted, not stored, so a read-horizon row is seeded as a failed read."""
    op_id = str(uuid.uuid4())
    await journal.log_write_op(write_op_id=op_id, operation=operation, kind=kind, success=success)
    created_at = (datetime.now(UTC) - timedelta(days=age_days)).isoformat()
    with contextlib.closing(sqlite3.connect(journal.db_path)) as conn:
        conn.execute('UPDATE write_ops SET created_at = ? WHERE id = ?', (created_at, op_id))
        conn.commit()
    return op_id


def _surviving_ids(journal: WriteJournal) -> set[str]:
    with contextlib.closing(sqlite3.connect(journal.db_path)) as conn:
        return {row[0] for row in conn.execute('SELECT id FROM write_ops')}


@pytest.mark.asyncio
async def test_pass_takes_every_horizon_from_the_config(journal):
    config = WriteJournalConfig(
        read_retention_days=10, search_retention_days=15, write_retention_days=15
    )
    rows = [('get_task', 'read', False), ('search', 'read', True), ('add_memory', 'write', True)]
    aged = {
        await _seed(journal, operation=op, kind=kind, success=ok, age_days=20)
        for op, kind, ok in rows
    }
    recent = {
        await _seed(journal, operation=op, kind=kind, success=ok, age_days=5)
        for op, kind, ok in rows
    }

    deleted = await prune_write_ops_pass(journal, config)

    assert deleted == 3, (
        f'expected the three 20-day rows deleted, got {deleted}; the shipped '
        'defaults (30/365/730) would delete none, so this pins the config mapping'
    )
    remaining = _surviving_ids(journal)
    assert aged.isdisjoint(remaining)
    assert recent <= remaining


@pytest.mark.asyncio
async def test_pass_takes_its_row_budget_from_the_config(journal):
    config = WriteJournalConfig(
        read_retention_days=10, prune_max_rows_per_run=1, prune_batch_size=1
    )
    for _ in range(2):
        await _seed(journal, operation='get_task', kind='read', success=False, age_days=20)

    assert await prune_write_ops_pass(journal, config) == 1
    assert len(_surviving_ids(journal)) == 1


@pytest.mark.asyncio
async def test_the_next_pass_resumes_a_backlog_the_row_budget_left(journal):
    config = WriteJournalConfig(
        read_retention_days=10, prune_max_rows_per_run=2, prune_batch_size=1
    )
    for _ in range(4):
        await _seed(journal, operation='get_task', kind='read', success=False, age_days=20)

    assert await prune_write_ops_pass(journal, config) == 2
    assert await prune_write_ops_pass(journal, config) == 2
    assert not _surviving_ids(journal), 'a pass resumed past rows its budget left behind'


@pytest.mark.asyncio
async def test_each_horizon_resumes_where_the_last_pass_drained_it(journal):
    config = WriteJournalConfig(read_retention_days=10)
    await _seed(journal, operation='get_task', kind='read', success=False, age_days=20)
    assert await prune_write_ops_pass(journal, config) == 1

    behind = await _seed(journal, operation='get_task', kind='read', success=False, age_days=30)

    assert await prune_write_ops_pass(journal, config) == 0, (
        'the second pass re-walked the range the first one drained; on the live '
        'journal that range holds every kept search row, 9.4 s cold in one statement'
    )
    assert behind in _surviving_ids(journal)

    restarted = WriteJournal(journal.data_dir)
    await restarted.initialize()
    try:
        assert await prune_write_ops_pass(restarted, config) == 1, (
            'a restarted journal must walk each horizon once from its oldest row'
        )
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_a_pass_without_the_kind_time_index_fails_loudly_instead_of_scanning(
    journal, caplog
):
    aged = await _seed(journal, operation='get_task', kind='read', success=False, age_days=60)
    with contextlib.closing(sqlite3.connect(journal.db_path)) as conn:
        conn.execute('DROP INDEX idx_wo_kind_time')

    with caplog.at_level(logging.ERROR, logger=write_journal.__name__):
        deleted = await prune_write_ops_pass(journal, WriteJournalConfig())

    assert deleted == 0 and aged in _surviving_ids(journal), (
        'the prune deleted without idx_wo_kind_time, so it scanned the table'
    )
    assert any(
        record.levelno == logging.ERROR and 'idx_wo_kind_time' in record.getMessage()
        for record in caplog.records
    ), 'the missing index must be named at ERROR'


@pytest.mark.asyncio
async def test_a_search_is_journalled_as_a_read_whatever_kind_it_is_given(journal, caplog):
    config = WriteJournalConfig(read_retention_days=10, search_retention_days=15)

    with caplog.at_level(logging.WARNING, logger=write_journal.__name__):
        mislabelled = await _seed(journal, operation='search', kind='write', age_days=20)

    assert await prune_write_ops_pass(journal, config) == 1, (
        'a search journalled as kind=write escaped the 15-day search horizon and '
        'would be kept until the 730-day write horizon'
    )
    assert mislabelled not in _surviving_ids(journal)
    assert any(
        mislabelled in record.getMessage() and "kind='write'" in record.getMessage()
        for record in caplog.records
    ), 'the override must name the write_op and the kind it was given'


class _RecordingJournal(WriteJournal):
    """Records each prune pass's bounds instead of touching a database."""

    def __init__(self, data_dir, *, fail_first: bool = False):
        super().__init__(data_dir)
        self.passes: list[dict] = []
        self.pass_ran = asyncio.Event()
        self._fail_next = fail_first

    async def prune_write_ops(
        self,
        *,
        read_older_than_days: float,
        search_older_than_days: float,
        write_older_than_days: float,
        batch_size: int,
        max_rows: int,
        max_seconds: float,
    ) -> int:
        self.passes.append({
            'read_older_than_days': read_older_than_days,
            'search_older_than_days': search_older_than_days,
            'write_older_than_days': write_older_than_days,
            'batch_size': batch_size,
            'max_rows': max_rows,
            'max_seconds': max_seconds,
        })
        self.pass_ran.set()
        if self._fail_next:
            self._fail_next = False
            raise RuntimeError('prune pass failed')
        return 0

    async def wait_for_passes(self, count: int) -> None:
        while len(self.passes) < count:
            self.pass_ran.clear()
            await self.pass_ran.wait()


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_loop_sleeps_before_its_first_pass(tmp_path):
    fake = _RecordingJournal(tmp_path)
    task = asyncio.create_task(
        run_periodic_write_ops_prune(fake, WriteJournalConfig(prune_interval_seconds=60))
    )

    await asyncio.sleep(0.05)

    assert fake.passes == [], 'the startup pass already ran; the loop must sleep first'
    await _cancel(task)


@pytest.mark.asyncio
async def test_loop_runs_a_pass_per_interval_until_cancelled(tmp_path):
    fake = _RecordingJournal(tmp_path)
    config = WriteJournalConfig(prune_interval_seconds=0.01)
    task = asyncio.create_task(run_periodic_write_ops_prune(fake, config))

    await asyncio.wait_for(fake.wait_for_passes(2), timeout=2.0)
    await _cancel(task)

    assert task.cancelled(), 'the loop must end cancelled, not swallow the cancel'
    expected = {
        'read_older_than_days': config.read_retention_days,
        'search_older_than_days': config.search_retention_days,
        'write_older_than_days': config.write_retention_days,
        'batch_size': config.prune_batch_size,
        'max_rows': config.prune_max_rows_per_run,
        'max_seconds': config.prune_max_seconds,
    }
    assert all(bounds == expected for bounds in fake.passes)


@pytest.mark.asyncio
async def test_a_failing_pass_does_not_stop_the_loop(tmp_path, caplog):
    fake = _RecordingJournal(tmp_path, fail_first=True)
    config = WriteJournalConfig(prune_interval_seconds=0.01)

    with caplog.at_level(logging.ERROR, logger=write_ops_retention.__name__):
        task = asyncio.create_task(run_periodic_write_ops_prune(fake, config))
        await asyncio.wait_for(fake.wait_for_passes(2), timeout=2.0)
        await _cancel(task)

    assert any(
        record.name == write_ops_retention.__name__ and record.levelno == logging.ERROR
        for record in caplog.records
    ), 'a failed pass must be logged at ERROR by the retention loop'
