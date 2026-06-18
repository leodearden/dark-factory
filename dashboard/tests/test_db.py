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

        All three cases return None pre-fix.  After the fix (``resolved.as_uri()``)
        all three open a real connection and ``SELECT 1`` returns ``(1,)``.
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

        # Pre-compute the URI the production code produces for path B so we can
        # match it exactly.  Using as_uri() here mirrors the production encoding
        # precisely, making the check robust to any tmp_path component that
        # as_uri() would percent-encode (spaces, special chars, etc.).
        b_uri_fragment = b_resolved.as_uri()

        async def wrapper(*args, **kwargs):
            nonlocal connect_calls
            connect_calls += 1
            uri_str = args[0] if args else ''
            if b_uri_fragment in uri_str:
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

    async def test_get_builds_sqlite_uri_via_path_as_uri(self, tmp_path):
        """DbPool.get must build the SQLite connect URI in canonical file:/// form.

        Verifies that the connect string passed to aiosqlite.connect starts with
        ``file:///``, ends with ``?mode=ro``, carries ``uri=True``, and that the
        path portion contains no unencoded ``?`` characters.  Exact formula is not
        re-derived (that would be a change-detector); the implementation-independent
        checks below fully enforce the URI canonicalization contract.
        """
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        real_connect = aiosqlite.connect
        captured_uri = None
        captured_uri_kwarg = None

        async def spy(*args, **kwargs):
            nonlocal captured_uri, captured_uri_kwarg
            captured_uri = args[0] if args else None
            captured_uri_kwarg = kwargs.get('uri')
            return await real_connect(*args, **kwargs)

        try:
            with patch('aiosqlite.connect', spy):
                conn = await pool.get(db_path)
            assert conn is not None, 'pool.get returned None — could not open db'
            assert captured_uri is not None
            assert captured_uri.startswith('file:///'), (
                f'Expected file:/// form (Path.as_uri()), got {captured_uri!r}'
            )
            assert captured_uri_kwarg is True, (
                f'Expected uri=True kwarg, got uri={captured_uri_kwarg!r}'
            )
            # Implementation-independent checks (don't mirror as_uri() formula):
            assert captured_uri.endswith('?mode=ro'), (
                f'URI must end with ?mode=ro, got {captured_uri!r}'
            )
            # Path portion must not contain a literal '?' (reserved chars must be %-encoded).
            path_portion = captured_uri.removesuffix('?mode=ro')
            assert '?' not in path_portion, (
                f'Unencoded ? in path portion of URI: {captured_uri!r}'
            )
            # Canonicalization check: path portion must equal the *resolved* db path URI.
            # If .resolve() were dropped from production, this would catch the regression on
            # platforms where db_path contains symlink components.
            resolved_db = db_path.resolve()
            assert path_portion == resolved_db.as_uri(), (
                f'URI path portion does not match resolved-path URI; '
                f'got {path_portion!r}, want {resolved_db.as_uri()!r}'
            )
        finally:
            await pool.close_all()

    async def test_get_warns_on_corrupt_or_locked_db(self, tmp_path, caplog):
        """DbPool.get must emit a WARNING when the file exists but connect fails.

        The benign 'not resolved.exists()' early-return stays silent; only a genuine
        OperationalError (corrupt file / exclusive lock) should produce a WARNING.
        """
        import logging

        db_path = tmp_path / 'corrupt.db'
        db_path.write_bytes(b'not a sqlite db')  # file exists, but corrupt

        pool = DbPool()
        with caplog.at_level(logging.WARNING, logger='dashboard.data.db'), patch('aiosqlite.connect', side_effect=sqlite3.OperationalError('file is not a database')):
            result = await pool.get(db_path)

        assert result is None
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == 'dashboard.data.db']
        assert warnings, 'expected a WARNING from dashboard.data.db, got none'

    async def test_get_no_warning_on_benign_missing_file(self, tmp_path, caplog):
        """DbPool.get must NOT emit a WARNING for a benign first-run absent file.

        The 'if not resolved.exists(): return None' early-return is silent by design.
        """
        import logging

        db_path = tmp_path / 'nonexistent' / 'missing.db'

        pool = DbPool()
        with caplog.at_level(logging.DEBUG, logger='dashboard.data.db'):
            result = await pool.get(db_path)

        assert result is None
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == 'dashboard.data.db']
        assert not warnings, f'expected no WARNING for benign missing file, got: {warnings}'

    async def test_close_all_clears_open_locks(self, tmp_path):
        """close_all() must drain _open_locks to prevent unbounded lock growth.

        A regression that drops the `_open_locks.clear()` line would pass CI
        silently without this assertion, allowing the lock map to grow
        unboundedly across distinct DB paths over a process lifetime.
        """
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        await pool.get(db_path)
        # Sanity check: opening a path should populate _open_locks.
        assert len(pool._open_locks) >= 1
        await pool.close_all()
        assert pool._open_locks == {}

    async def test_get_returns_none_after_close_all(self, tmp_path):
        """get() must return None once close_all() has been called.

        Guards the close/open race: a concurrent get() that is mid-open when
        close_all() runs must see _closed=True after acquiring the per-path lock
        and abort cleanly rather than installing a connection into a drained pool.
        This sequential version pins the basic invariant without needing concurrency.
        """
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        conn = await pool.get(db_path)
        assert conn is not None
        await pool.close_all()
        # Pool is now closed; a new get() must not re-open or leak a connection.
        result = await pool.get(db_path)
        assert result is None
        # open_count stays at 0 — no connection was installed post-close.
        assert pool.open_count == 0

    async def test_get_suspended_in_connect_during_close_all_does_not_leak(
        self, tmp_path
    ):
        """get() suspended inside aiosqlite.connect() when close_all() runs must
        close the freshly-opened connection and return None — not install a stranded
        connection into the drained pool.

        Race window: get() acquires the per-path lock, passes the pre-connect
        `if self._closed` check (False at that point), then suspends inside
        `await aiosqlite.connect()`.  close_all() runs to completion — sets
        _closed=True, drains _conns, clears _open_locks — WITHOUT holding the
        per-path lock.  get() then resumes.

        Pre-fix: get() has no post-connect re-check → installs the connection,
        returns it (not None), open_count==1, and leaves the aiosqlite worker
        thread stranded (result is not None, open_count != 0 → RED).

        Post-fix: `if self._closed: await conn.close(); return None` after
        aiosqlite.connect() → result is None, open_count==0, and the
        mid-flight connection is closed (use-after-close raises) → GREEN.
        """
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        real_connect = aiosqlite.connect
        # Track every aiosqlite.Connection object opened inside the wrapper so
        # we can verify the mid-flight connection was actually closed.
        opened: list[aiosqlite.Connection] = []

        # Two events for deterministic synchronisation:
        #   inside_connect  — get() has entered aiosqlite.connect() (suspended)
        #   close_done      — close_all() has completed
        inside_connect = asyncio.Event()
        close_done = asyncio.Event()

        async def wrapper(*args, **kwargs):
            inside_connect.set()          # signal: get() is now suspended mid-open
            await close_done.wait()       # wait until close_all() has finished
            conn = await real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        with patch('aiosqlite.connect', wrapper):
            # Launch get() as a background task; it will suspend inside wrapper.
            getter = asyncio.create_task(pool.get(db_path))

            # Wait until get() is suspended inside aiosqlite.connect().
            await asyncio.wait_for(inside_connect.wait(), timeout=2.0)

            # close_all() runs now: sets _closed, drains _conns, clears
            # _open_locks — it does NOT hold the per-path lock, so getter is
            # still suspended in wrapper above.
            await pool.close_all()

            # Unblock get() so it can resume after aiosqlite.connect() returns.
            close_done.set()

            result = await asyncio.wait_for(getter, timeout=2.0)

        # get() must return None — connection must NOT be installed post-close.
        assert result is None, (
            f'expected None (post-connect _closed re-check), got {result!r}'
        )
        # The pool must remain empty — no stranded connection.
        assert pool.open_count == 0, (
            f'expected open_count=0, got {pool.open_count}'
        )

        # The mid-flight connection was physically opened (wrapper appended it).
        assert len(opened) == 1, f'expected 1 opened connection, got {len(opened)}'

        # The post-connect re-check must have CLOSED the mid-flight connection so
        # its aiosqlite worker thread is not stranded.  Assert on observable state
        # rather than `pytest.raises(...)` on a subsequent execute: which exception
        # aiosqlite raises for use-after-close has shifted between versions
        # (ValueError, sqlite3.ProgrammingError, RuntimeError, OperationalError,
        # aiosqlite.Error have all been observed), and a stranded worker thread can
        # hang on `await` rather than raise — defeating any exception-based check.
        # Both flags are set synchronously inside `await conn.close()`, which runs
        # in the resumed getter task — they are therefore settled by the time
        # `await getter` returns above.
        # Verified against aiosqlite >=0.22.x — bump and re-verify the
        # private-attribute lifecycle if this pin moves.
        assert (
            hasattr(opened[0], '_connection')
            and hasattr(opened[0], '_running')
            and hasattr(opened[0], '_thread')
        ), 'aiosqlite internal attribute names changed — update test'
        assert opened[0]._connection is None, (
            f'expected closed mid-flight connection (_connection is None), '
            f'got {opened[0]._connection!r}'
        )
        assert opened[0]._running is False, (
            f'expected worker-thread shutdown (_running is False), '
            f'got {opened[0]._running!r}'
        )
        # Defense-in-depth: confirm the worker thread itself terminated, not just
        # that the state flags were flipped.  The thread exits after the STOP
        # sentinel is processed — which happens asynchronously after close()
        # returns — so join(timeout=2.0) is required before asserting is_alive().
        opened[0]._thread.join(timeout=2.0)
        assert not opened[0]._thread.is_alive(), (
            'aiosqlite worker thread did not exit after close '
            '(mid-open connection was not properly closed by the '
            'post-connect _closed re-check)'
        )


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

    async def test_with_db_warns_on_operational_error(self, tmp_path, caplog):
        """with_db must emit a WARNING when fn raises sqlite3.OperationalError.

        The return contract is unchanged: the supplied default is returned.
        Fails today because with_db logs at DEBUG, not WARNING.
        """
        import logging

        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        async with aiosqlite.connect(str(db_path)) as db:

            async def bad_query(conn):
                raise sqlite3.OperationalError('no such table: nonexistent')

            with caplog.at_level(logging.WARNING, logger='dashboard.data.db'):
                result = await with_db(db, bad_query, 'fallback')

        assert result == 'fallback'
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == 'dashboard.data.db']
        assert warnings, 'expected a WARNING from dashboard.data.db on query failure, got none'

    async def test_with_db_warns_on_os_error(self, tmp_path, caplog):
        """with_db must emit a WARNING when fn raises OSError.

        The return contract is unchanged: the supplied default is returned.
        Fails today because with_db logs at DEBUG, not WARNING.
        """
        import logging

        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        async with aiosqlite.connect(str(db_path)) as db:

            async def raises_os_error(conn):
                raise OSError('disk I/O error')

            with caplog.at_level(logging.WARNING, logger='dashboard.data.db'):
                result = await with_db(db, raises_os_error, 42)

        assert result == 42
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == 'dashboard.data.db']
        assert warnings, 'expected a WARNING from dashboard.data.db on OSError, got none'
