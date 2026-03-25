"""Tests for dashboard.data.db — DbPool and with_db helper."""

from __future__ import annotations

import sqlite3

import aiosqlite

from dashboard.data.db import DbPool, with_db


class TestDbPool:
    """Tests for the DbPool connection cache."""

    async def test_get_opens_connection(self, tmp_path):
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        conn = await pool.get(db_path)
        assert conn is not None
        assert isinstance(conn, aiosqlite.Connection)
        await pool.close_all()

    async def test_get_missing_returns_none(self, tmp_path):
        pool = DbPool()
        conn = await pool.get(tmp_path / 'nonexistent' / 'nope.db')
        assert conn is None

    async def test_get_reuses_connection(self, tmp_path):
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        conn1 = await pool.get(db_path)
        conn2 = await pool.get(db_path)
        assert conn1 is conn2
        await pool.close_all()

    async def test_close_all(self, tmp_path):
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        await pool.get(db_path)
        assert pool.open_count == 1
        await pool.close_all()
        assert pool.open_count == 0

    async def test_open_count(self, tmp_path):
        pool = DbPool()
        assert pool.open_count == 0

        for i in range(3):
            p = tmp_path / f'db{i}.db'
            sqlite3.connect(str(p)).close()
            await pool.get(p)

        assert pool.open_count == 3
        await pool.close_all()


class TestWithDb:
    """Tests for the with_db helper."""

    async def test_returns_result_on_success(self, tmp_path):
        db_path = tmp_path / 'test.db'
        conn = sqlite3.connect(str(db_path))
        conn.execute('CREATE TABLE t (x)')
        conn.execute('INSERT INTO t VALUES (42)')
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as db:
            async def query(db):
                async with db.execute('SELECT x FROM t') as cur:
                    row = await cur.fetchone()
                return row[0]

            result = await with_db(db, query, -1)
            assert result == 42

    async def test_returns_default_on_none(self):
        result = await with_db(None, lambda db: db.execute('SELECT 1'), 'default')
        assert result == 'default'

    async def test_returns_default_on_operational_error(self, tmp_path):
        db_path = tmp_path / 'empty.db'
        sqlite3.connect(str(db_path)).close()  # no tables

        async with aiosqlite.connect(str(db_path)) as db:
            async def bad_query(db):
                async with db.execute('SELECT * FROM nonexistent_table') as cur:
                    return await cur.fetchall()

            result = await with_db(db, bad_query, [])
            assert result == []
