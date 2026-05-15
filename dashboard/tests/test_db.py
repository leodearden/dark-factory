"""Tests for dashboard.data.db — DbPool and with_db helper."""

from __future__ import annotations

import asyncio
import sqlite3
from unittest.mock import patch

import aiosqlite
import pytest

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

    async def test_get_returns_none_on_connect_os_error(self, tmp_path):
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        with patch('aiosqlite.connect', side_effect=OSError('disk error')):
            result = await pool.get(db_path)
        assert result is None

    async def test_get_returns_none_on_permission_error(self, tmp_path):
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        with patch('pathlib.Path.exists', side_effect=PermissionError('permission denied')):
            result = await pool.get(db_path)
        assert result is None

    @pytest.mark.parametrize('fragment', ['has?question', 'has#hash', 'has%41literal'])
    async def test_get_handles_uri_reserved_chars_in_path(self, tmp_path, fragment):
        """Regression canary for task 1337 (mirrors task 1331 in test_scheduler_state_tools.py).

        DbPool.get must URL-encode the filesystem path before building the SQLite URI
        (``file:<path>?mode=ro``).  Three URI-reserved characters are tested:

        - ``?``: URI query-string delimiter — SQLite truncates the path at ``?``,
          targeting a non-existent file; DbPool.get catches OperationalError and
          returns None.
        - ``#``: URI fragment delimiter — SQLite truncates the path at ``#``;
          same result.
        - ``%41``: literal percent-sequence — SQLite decodes it to ``A``, silently
          targeting a different path that does not exist.

        All three cases return None pre-fix.  After the fix (``quote(str(resolved),
        safe='/')``) all three open a real connection and ``SELECT 1`` returns ``(1,)``.
        """
        subdir = tmp_path / fragment
        subdir.mkdir()
        db_path = subdir / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        try:
            conn = await pool.get(db_path)
            assert conn is not None, f'pool.get returned None for path containing {fragment!r}'
            async with conn.execute('SELECT 1') as cur:
                row = await cur.fetchone()
            assert row is not None
            assert row[0] == 1
        finally:
            await pool.close_all()

    async def test_concurrent_get_same_path_opens_once(self, tmp_path):
        """Concurrent get() calls for the same path must open exactly one connection.

        Pre-fix: all N coroutines pass the membership check before the first
        connect resolves, so aiosqlite.connect is called N times and N-1
        connections are leaked.  Post-fix: only one connection is opened; the
        rest reuse the cached result.
        """
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        real_connect = aiosqlite.connect
        connect_calls = 0

        async def wrapper(*args, **kwargs):
            nonlocal connect_calls
            connect_calls += 1
            # Yield so every gathered coroutine can pass the membership check
            # before the first connect resolves — this makes the race
            # deterministic and observable.
            await asyncio.sleep(0)
            return await real_connect(*args, **kwargs)

        try:
            with patch('aiosqlite.connect', wrapper):
                results = await asyncio.gather(*[pool.get(db_path) for _ in range(8)])
            assert connect_calls == 1, (
                f'expected 1 connect call, got {connect_calls} — '
                'concurrent gets opened duplicate connections'
            )
            assert all(r is not None for r in results), 'all results should be non-None'
            assert all(r is results[0] for r in results), (
                'all results should be the same cached connection object'
            )
            assert pool.open_count == 1
        finally:
            await pool.close_all()

    async def test_disjoint_paths_do_not_serialize(self, tmp_path):
        """Concurrent get() calls for *different* paths must not deadlock.

        A pool-wide lock causes deadlock when path A's connect waits for path B
        to connect (via an asyncio.Event) while B is blocked waiting for the
        same pool-wide lock that A holds.

        Pre-fix (step-2, pool-wide lock): asyncio.wait_for raises TimeoutError.
        Post-fix (step-4, per-path locks): both connections open concurrently,
        each using its own independent lock.
        """
        a_path = tmp_path / 'a.db'
        b_path = tmp_path / 'b.db'
        sqlite3.connect(str(a_path)).close()
        sqlite3.connect(str(b_path)).close()

        pool = DbPool()
        real_connect = aiosqlite.connect
        b_resolved = b_path.resolve()
        event_b_done = asyncio.Event()
        connect_calls = 0

        async def wrapper(*args, **kwargs):
            nonlocal connect_calls
            connect_calls += 1
            uri_str = args[0] if args else ''
            if str(b_resolved) in uri_str:
                # Path B: connect immediately, then signal path A to proceed.
                conn = await real_connect(*args, **kwargs)
                event_b_done.set()
                return conn
            else:
                # Path A: wait for B to connect first, then connect.
                # With a pool-wide lock A holds the lock here and B can never
                # connect → deadlock → TimeoutError (RED).
                await event_b_done.wait()
                return await real_connect(*args, **kwargs)

        try:
            with patch('aiosqlite.connect', wrapper):
                results = await asyncio.wait_for(
                    asyncio.gather(pool.get(a_path), pool.get(b_path)),
                    timeout=2.0,
                )
            assert results[0] is not None
            assert results[1] is not None
            assert results[0] is not results[1], 'disjoint paths must yield distinct connections'
            assert pool.open_count == 2
            assert connect_calls == 2
        finally:
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

    async def test_returns_default_on_os_error(self, tmp_path):
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        async with aiosqlite.connect(str(db_path)) as db:
            async def raises_os_error(db):
                raise OSError('disk I/O error')

            result = await with_db(db, raises_os_error, 'default')
            assert result == 'default'
