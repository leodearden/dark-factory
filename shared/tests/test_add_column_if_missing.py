"""Tests for shared.sqlite_column_migration, against real temp SQLite files.

The concurrent-adder race is reproduced, not faked: a ``sqlite3.Connection``
subclass handed in through the public ``factory=`` kwarg adds the column over a
SECOND real connection immediately before its own first ALTER executes, so
SQLite raises its genuine "duplicate column name".
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from shared.sqlite_column_migration import add_column_if_missing

_BAD_IDENTIFIERS = ['t; DROP TABLE t', 'c d', '']


def _create_table(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute('CREATE TABLE t (a INTEGER)')
        conn.execute('INSERT INTO t (a) VALUES (1)')
        conn.commit()
    finally:
        conn.close()


def _columns(path: Path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [row[1] for row in conn.execute('PRAGMA table_info(t)')]
    finally:
        conn.close()


def _recording_factory() -> tuple[type[sqlite3.Connection], list[str]]:
    seen: list[str] = []

    class RecordingConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
            seen.append(sql)
            return super().execute(sql, parameters)

    return RecordingConnection, seen


def _racing_factory(path: Path, column_ddl_sql: str) -> type[sqlite3.Connection]:
    raced = False

    class RacingConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
            nonlocal raced
            if not raced and sql.lstrip().upper().startswith('ALTER TABLE'):
                raced = True
                rival = sqlite3.connect(path)
                try:
                    rival.execute(column_ddl_sql)
                    rival.commit()
                finally:
                    rival.close()
            return super().execute(sql, parameters)

    return RacingConnection


def _holding_write_lock(path: Path) -> Callable[[], None]:
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute('BEGIN IMMEDIATE')

    def release() -> None:
        try:
            holder.execute('ROLLBACK')
        finally:
            holder.close()

    return release


class TestAddColumnIfMissingAsync:
    async def test_fresh_add_returns_true_and_applies_ddl_tail(self, tmp_path: Path) -> None:
        path = tmp_path / 'fresh.db'
        _create_table(path)

        async with aiosqlite.connect(path) as db:
            added = await add_column_if_missing(db, 't', 'c', 'INTEGER NOT NULL DEFAULT 7')
            await db.commit()
            rows = await db.execute_fetchall('SELECT c FROM t')

        assert added is True
        assert _columns(path) == ['a', 'c']
        assert [tuple(row) for row in rows] == [(7,)]

    async def test_present_column_returns_false_without_alter(self, tmp_path: Path) -> None:
        path = tmp_path / 'present.db'
        _create_table(path)
        async with aiosqlite.connect(path) as db:
            await add_column_if_missing(db, 't', 'c', 'TEXT')
            await db.commit()

        recording, seen = _recording_factory()
        async with aiosqlite.connect(path, factory=recording) as db:
            added = await add_column_if_missing(db, 't', 'c', 'TEXT')

        assert added is False
        assert _columns(path).count('c') == 1
        assert not [sql for sql in seen if sql.lstrip().upper().startswith('ALTER')]

    async def test_concurrent_adder_is_tolerated(self, tmp_path: Path) -> None:
        path = tmp_path / 'race.db'
        _create_table(path)
        racing = _racing_factory(path, 'ALTER TABLE t ADD COLUMN c TEXT')

        async with aiosqlite.connect(path, factory=racing) as db:
            added = await add_column_if_missing(db, 't', 'c', 'TEXT')

        assert added is False
        assert _columns(path).count('c') == 1

    async def test_missing_table_propagates(self, tmp_path: Path) -> None:
        path = tmp_path / 'missing.db'
        _create_table(path)

        async with aiosqlite.connect(path) as db:
            with pytest.raises(sqlite3.OperationalError, match='no such table'):
                await add_column_if_missing(db, 'missing', 'c', 'TEXT')

    async def test_locked_database_propagates(self, tmp_path: Path) -> None:
        path = tmp_path / 'locked.db'
        _create_table(path)
        release = _holding_write_lock(path)
        try:
            async with aiosqlite.connect(path, timeout=0) as db:
                with pytest.raises(sqlite3.OperationalError, match='database is locked'):
                    await add_column_if_missing(db, 't', 'c', 'TEXT')
        finally:
            release()

        assert 'c' not in _columns(path)

    @pytest.mark.parametrize('bad', _BAD_IDENTIFIERS)
    async def test_non_identifier_table_is_rejected_before_any_sql(
        self, tmp_path: Path, bad: str
    ) -> None:
        path = tmp_path / 'bad_table.db'
        _create_table(path)
        recording, seen = _recording_factory()

        async with aiosqlite.connect(path, factory=recording) as db:
            with pytest.raises(ValueError, match=r'^table\b') as excinfo:
                await add_column_if_missing(db, bad, 'c', 'TEXT')

        assert repr(bad) in str(excinfo.value)
        assert seen == []
        assert _columns(path) == ['a']

    @pytest.mark.parametrize('bad', _BAD_IDENTIFIERS)
    async def test_non_identifier_column_is_rejected_before_any_sql(
        self, tmp_path: Path, bad: str
    ) -> None:
        path = tmp_path / 'bad_column.db'
        _create_table(path)
        recording, seen = _recording_factory()

        async with aiosqlite.connect(path, factory=recording) as db:
            with pytest.raises(ValueError, match=r'^column\b') as excinfo:
                await add_column_if_missing(db, 't', bad, 'TEXT')

        assert repr(bad) in str(excinfo.value)
        assert seen == []
        assert _columns(path) == ['a']
