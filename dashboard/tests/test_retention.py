"""Tests for dashboard.data.retention — the snapshot tables' retention policy, and what it never touches."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest
from shared.quality_series.landings import GitCrossCheck, Landing, LandingSource, LinesBase
from shared.quality_series.rows import Cost, LandingRow, Outcomes, SourceEpochs

from dashboard.data.burndown import BURNDOWN_SCHEMA, downsample
from dashboard.data.metrics import METRICS_SCHEMA, downsample_metrics
from dashboard.data.quality_store import (
    QUALITY_SCHEMA,
    QualitySample,
    insert_sample,
    read_landing_rows,
    read_samples,
    upsert_landing_rows,
)
from dashboard.data.retention import (
    MAX_RETENTION,
    RAW_RETENTION,
    SqlPredicate,
    apply_retention,
)

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
OLD_HOUR = (NOW - timedelta(days=10)).replace(minute=0, second=0, microsecond=0)
EARLIER_OLD_HOUR = OLD_HOUR - timedelta(hours=1)

Row = tuple[str | None, str, str | None]


def _at(hour: datetime, minute: int) -> str:
    return (hour + timedelta(minutes=minute)).isoformat()


def _create_table(db_path: Path, rows: list[Row]) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        'CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,'
        ' project_id TEXT, kind TEXT)'
    )
    conn.executemany('INSERT INTO t (project_id, ts, kind) VALUES (?, ?, ?)', rows)
    conn.commit()
    conn.close()


async def _retain(
    db_path: Path,
    *,
    partition_by: tuple[str, ...] = (),
    prefer: SqlPredicate | None = None,
) -> None:
    async with aiosqlite.connect(str(db_path)) as conn:
        await apply_retention(conn, 't', now=NOW, partition_by=partition_by, prefer=prefer)
        await conn.commit()


def _survivors(db_path: Path) -> list[Row]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute('SELECT project_id, ts, kind FROM t ORDER BY ts, project_id').fetchall()
    finally:
        conn.close()


def _two_projects_two_old_hours() -> list[Row]:
    return [
        (project, _at(hour, minute), None)
        for project in ('a', 'b')
        for hour in (EARLIER_OLD_HOUR, OLD_HOUR)
        for minute in (10, 50)
    ]


async def test_keeps_the_newest_row_of_each_old_hour_per_partition(tmp_path):
    """Regression for task 6606: a constant bucket key would keep one row per project."""
    db_path = tmp_path / 'retention.db'
    _create_table(db_path, _two_projects_two_old_hours())

    await _retain(db_path, partition_by=('project_id',))

    expected = [
        (project, _at(hour, 50), None)
        for hour in (EARLIER_OLD_HOUR, OLD_HOUR)
        for project in ('a', 'b')
    ]
    got = _survivors(db_path)
    assert got == expected, f'expected {expected}, got {got}'


async def test_without_partition_columns_keeps_one_row_per_old_hour(tmp_path):
    db_path = tmp_path / 'retention.db'
    _create_table(db_path, _two_projects_two_old_hours())

    await _retain(db_path)

    got = _survivors(db_path)
    assert [ts for _, ts, _ in got] == [_at(EARLIER_OLD_HOUR, 50), _at(OLD_HOUR, 50)], (
        f'expected the newest row of each old hour across projects, got {got}'
    )


async def test_rows_inside_the_raw_window_are_untouched(tmp_path):
    db_path = tmp_path / 'retention.db'
    edge_hour = (NOW - (RAW_RETENTION - timedelta(hours=1))).replace(
        minute=0, second=0, microsecond=0,
    )
    recent_hour = (NOW - timedelta(minutes=10)).replace(minute=0, second=0, microsecond=0)
    rows: list[Row] = [
        ('a', _at(hour, minute), None)
        for hour in (edge_hour, recent_hour)
        for minute in (1, 2, 3)
    ]
    _create_table(db_path, rows)

    await _retain(db_path, partition_by=('project_id',))

    got = _survivors(db_path)
    assert got == sorted(rows, key=lambda row: row[1]), f'expected every raw row, got {got}'


async def test_rows_older_than_max_retention_are_deleted(tmp_path):
    db_path = tmp_path / 'retention.db'
    expired = (NOW - MAX_RETENTION - timedelta(hours=1)).isoformat()
    old_but_kept = (NOW - MAX_RETENTION + timedelta(days=1)).isoformat()
    _create_table(db_path, [('a', expired, None), ('a', old_but_kept, None)])

    await _retain(db_path, partition_by=('project_id',))

    got = _survivors(db_path)
    assert got == [('a', old_but_kept, None)], (
        f'expected only the row inside {MAX_RETENTION}, got {got}'
    )


async def test_a_preferred_row_outranks_a_newer_one_in_its_hour(tmp_path):
    db_path = tmp_path / 'retention.db'
    _create_table(db_path, [
        ('a', _at(EARLIER_OLD_HOUR, 10), 'gap'),
        ('a', _at(EARLIER_OLD_HOUR, 50), 'gap'),
        ('a', _at(OLD_HOUR, 10), 'value'),
        ('a', _at(OLD_HOUR, 50), 'gap'),
    ])

    await _retain(
        db_path, partition_by=('project_id',), prefer=SqlPredicate('kind = ?', ('value',)),
    )

    expected = [
        ('a', _at(EARLIER_OLD_HOUR, 50), 'gap'),
        ('a', _at(OLD_HOUR, 10), 'value'),
    ]
    got = _survivors(db_path)
    assert got == expected, f'expected {expected}, got {got}'


async def test_a_partition_column_that_is_not_an_identifier_is_refused(tmp_path):
    db_path = tmp_path / 'retention.db'
    rows = _two_projects_two_old_hours()
    _create_table(db_path, rows)

    with pytest.raises(ValueError, match=r"'t'.*'project id'"):
        await _retain(db_path, partition_by=('project id',))

    got = _survivors(db_path)
    assert got == sorted(rows, key=lambda row: (row[1], row[0])), (
        f'expected the refused call to leave every row, got {got}'
    )


async def test_a_table_name_that_is_not_an_identifier_is_refused(tmp_path):
    db_path = tmp_path / 'retention.db'
    _create_table(db_path, [])

    async with aiosqlite.connect(str(db_path)) as conn:
        with pytest.raises(ValueError, match=r"'t; DROP TABLE t'"):
            await apply_retention(conn, 't; DROP TABLE t', now=NOW)


async def test_quality_tables_survive_every_retention_pass(tmp_path):
    """quality.db is outside the retention policy: neither caller of apply_retention touches it.

    PRD decision 14 (plans/quality-dashboard-prd.md). The samples sit in old hours that
    hold two each, one hour past MAX_RETENTION and one inside the compaction window, so
    either an expiry or a compaction pass over a quality table would change what reads back.
    """
    expired_hour = (NOW - timedelta(days=200)).replace(minute=0, second=0, microsecond=0)
    landing = Landing(
        project_id='p', merge_sha='sha', task_id='101', landed_at=expired_hour, ordinal=0,
        source=LandingSource.MERGE_FINALIZED, unit_id='sha', unit_size=1, late_claim=False,
        git_first_parent=True, prev_landed_at=None, lines_base=LinesBase.UNMEASURABLE,
        base_sha=None,
    )
    row = LandingRow(
        landing=landing,
        outcomes=Outcomes(None, None, None, None, None, None, None, None, None),
        cost=Cost(None, (), None, None, None, False, None, None, None, None, None, None, None),
        modules=(),
        rule_vintage=SourceEpochs(None, None, None, None, None, None),
    )
    samples = [
        QualitySample.value(
            'p', hour + timedelta(minutes=minute), cursor_sha='sha', landings_sampled=1,
            landings_deferred=0, git_check=GitCrossCheck(0, 1, 0),
        )
        for hour in (expired_hour, OLD_HOUR)
        for minute in (10, 50)
    ]
    async with aiosqlite.connect(str(tmp_path / 'combined.db')) as conn:
        await conn.executescript(BURNDOWN_SCHEMA + METRICS_SCHEMA + QUALITY_SCHEMA)
        await upsert_landing_rows(conn, [row], sampled_at=expired_hour)
        for sample in samples:
            await insert_sample(conn, sample)
        await conn.commit()

        await downsample(conn, now=NOW)
        await downsample_metrics(conn, now=NOW)

        assert await read_landing_rows(conn, 'p') == [row]
        assert await read_samples(conn, 'p') == samples
