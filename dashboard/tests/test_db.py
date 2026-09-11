"""Tests for dashboard.data.db — DbPool and with_db helper."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
from collections.abc import Callable
from unittest.mock import patch

import aiosqlite
import pytest
from _dashboard_helpers import live_aiosqlite_worker_threads

from dashboard.data.db import DbPool, with_db


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    """Poll *predicate* until it is true, or fail the enclosing wait_for.

    Used to await a done-callback that runs on the event loop but that the test
    holds no future for (``DbPool``'s in-flight connect bookkeeping).  Polling
    rather than sleeping a fixed interval keeps the tests race-free.
    """

    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=timeout)


async def _reap(conns: list[aiosqlite.Connection]) -> None:
    """Close any still-running connection so a RED run does not wedge pytest.

    ``DbPool`` opens via plain ``aiosqlite.connect``, whose worker thread
    inherits ``daemon=False`` from the main thread.  A connection these tests
    deliberately strand therefore blocks interpreter exit — the pytest process
    hangs AFTER writing its report.  Every test below that can strand one calls
    this from a ``finally``.
    """
    for conn in conns:
        if getattr(conn, '_running', False):
            with contextlib.suppress(Exception):
                await conn.close()


class TestLiveAiosqliteWorkerThreads:
    """Self-check for the leak detector every thread assertion below depends on.

    ``live_aiosqlite_worker_threads()`` keys on TWO private attributes —
    CPython's ``threading.Thread._target`` and aiosqlite's module-level
    ``_connection_worker_thread`` — and reads both through
    ``getattr(..., None)``.  If either moves it returns ``[]``, and every leak
    assertion built on it (``survivors`` here, ``leaked`` in
    ``test_durability.py``) passes VACUOUSLY: the exact silent-fail-soft the
    detector exists to prevent.  ``Thread._target``'s lifetime is already
    implementation-coupled — ``Thread.run()`` deletes it in its ``finally`` —
    so this is not a hypothetical.

    A positive round-trip against a REAL connection pins both attributes at
    once and cannot degrade to a quiet pass, which a ``hasattr`` guard on one
    of the two names could (and, being a bare module-level ``assert``, was also
    stripped under ``python -O``).
    """

    async def test_detects_and_then_stops_detecting_a_real_connection(self):
        before = set(live_aiosqlite_worker_threads())
        conn = await aiosqlite.connect(':memory:')
        try:
            found = [t for t in live_aiosqlite_worker_threads() if t not in before]
            assert len(found) == 1, (
                f'live_aiosqlite_worker_threads() found {len(found)} new worker '
                f'thread(s) for one open aiosqlite connection, expected 1 — the '
                f'private-attribute pin (Thread._target / '
                f'aiosqlite.core._connection_worker_thread) no longer matches '
                f'aiosqlite {aiosqlite.__version__}, so every leak assertion in '
                f'this file and test_durability.py is now vacuous'
            )
            assert found[0] is conn._thread, (
                f'the detected worker is not this connection\'s thread '
                f'({found[0]!r} is not {conn._thread!r})'
            )
        finally:
            await conn.close()

        conn._thread.join(timeout=2.0)
        assert not conn._thread.is_alive(), 'worker thread did not exit after close()'
        still = [t for t in live_aiosqlite_worker_threads() if t not in before]
        assert not still, (
            f'live_aiosqlite_worker_threads() still reports {len(still)} worker '
            f'thread(s) after close() — the detector reports dead threads as '
            f'live, so leak assertions would fire spuriously'
        )


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
        self, tmp_path, monkeypatch
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
        # This test holds its connect open until AFTER close_all() returns, so
        # close_all()'s in-flight drain (task 3466) necessarily waits out its
        # full budget.  Shrink the budget rather than pay the production 5s on
        # every suite run; the choreography and assertions are unaffected.
        monkeypatch.setattr('dashboard.data.db._INFLIGHT_DRAIN_TIMEOUT', 0.05)
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

    async def test_cancelled_get_adopts_connection_into_pool(self, tmp_path):
        """A get() cancelled mid-connect must hand the landing connection to the pool.

        This is the production ``/healthz`` abandoned-probe path: ``_probe_db``
        (dashboard/app.py) cancels its probe task at the deadline and does NOT
        await the unwinding, so ``pool.get()`` can be cancelled while suspended
        inside ``aiosqlite.connect()`` with the pool still very much OPEN.

        Distinct from the close_all() race above: nothing is draining the pool
        here, so ``close_all()``'s in-flight drain can never see this connect.
        The only place ownership can land is the connect task's own
        done-callback.

        Pre-fix (step-2 only): the shielded connect runs to completion and its
        ``Connection`` is dropped on the floor — open_count stays 0, the worker
        thread lives on, and the next get() for the same path opens a SECOND
        connection → RED.

        Post-fix: the pool ADOPTS the landed connection into ``_conns``, so it
        is reused by the next get() and reaped by close_all() → GREEN.
        """
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        real_connect = aiosqlite.connect
        opened: list[aiosqlite.Connection] = []

        # Same two-Event choreography as the close_all race test above.
        inside_connect = asyncio.Event()
        resume = asyncio.Event()

        async def wrapper(*args, **kwargs):
            inside_connect.set()      # signal: get() is now suspended mid-open
            await resume.wait()       # hold until the test has cancelled get()
            conn = await real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        try:
            with patch('aiosqlite.connect', wrapper):
                getter = asyncio.create_task(pool.get(db_path))
                await asyncio.wait_for(inside_connect.wait(), timeout=2.0)

                # The interposed event: cancel the CALLER (not close_all).
                getter.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await getter

                # Let the shielded connect land, now that nobody is awaiting it.
                resume.set()
                await _wait_until(lambda: pool.inflight_count == 0)

                assert len(opened) == 1, (
                    f'expected exactly 1 physical connect, got {len(opened)}'
                )

                # (1) The landed connection was ADOPTED, not discarded.
                assert pool.open_count == 1, (
                    f'expected open_count=1 (connection adopted after the '
                    f'getter was cancelled), got {pool.open_count} — the '
                    f'connect landed with nobody owning it and its worker '
                    f'thread is now stranded'
                )
                reused = await pool.get(db_path)
                assert reused is opened[0], (
                    f'expected the adopted connection to be reused, got '
                    f'{reused!r} (is opened[0]: {reused is opened[0]})'
                )
                assert len(opened) == 1, (
                    f'a second aiosqlite.connect() happened ({len(opened)} '
                    f'total) — the adopted connection was not reused'
                )

            # (2) close_all() reaps the adopted connection and its worker thread.
            await pool.close_all()
            assert pool.open_count == 0, (
                f'expected open_count=0 after close_all, got {pool.open_count}'
            )
            # Verified against aiosqlite >=0.22.x — see the private-attribute
            # pin rationale on the close_all race test above.
            assert (
                hasattr(opened[0], '_connection')
                and hasattr(opened[0], '_running')
                and hasattr(opened[0], '_thread')
            ), 'aiosqlite internal attribute names changed — update test'
            assert opened[0]._connection is None, (
                f'expected closed adopted connection (_connection is None), '
                f'got {opened[0]._connection!r}'
            )
            assert opened[0]._running is False, (
                f'expected worker-thread shutdown (_running is False), '
                f'got {opened[0]._running!r}'
            )
            opened[0]._thread.join(timeout=2.0)
            assert not opened[0]._thread.is_alive(), (
                'aiosqlite worker thread did not exit after close_all — the '
                'connection adopted from the cancelled getter was not reaped'
            )
        finally:
            await _reap(opened)

    async def test_racing_getter_does_not_overwrite_adopted_connection(self, tmp_path):
        """An ADOPTED connection must survive a getter that was already in flight.

        ``_adopt_or_close`` installs into ``_conns`` WITHOUT holding
        ``_open_locks[path]`` — the cancelled getter released that lock while
        unwinding through ``async with lock``, so a second getter for the same
        path can be suspended inside its own ``aiosqlite.connect()`` when the
        adoption happens.

        Pre-fix: ``get()``'s post-shield re-check tests only ``self._closed``,
        never ``resolved in self._conns``, so the second getter's connection
        OVERWRITES the adopted one.  The adopted connection is then in neither
        ``_conns`` nor ``_inflight`` — ``close_all()`` cannot reach it, and its
        (non-daemon) aiosqlite worker thread lives for the process lifetime.

        Post-fix: ``get()`` keeps the incumbent, closes the connection it just
        opened, and returns the pooled one — the same one-connection-per-path
        contract already pinned by ``test_get_reuses_connection`` and
        ``test_concurrent_get_same_path_opens_once``.

        Production reachability is real: ``/healthz``'s ``_probe_db``
        (dashboard/app.py) is the cancelling getter and ``_metrics_loop``'s
        ``_run_once`` gets the very same paths on its poll cycle.
        """
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        real_connect = aiosqlite.connect
        opened: list[aiosqlite.Connection] = []

        # Per-connect gates: connect #1 is A's, connect #2 is B's.  Counting
        # invocations (rather than inspecting args) is what lets one wrapper
        # drive two independently-released connects for the SAME path.
        calls = 0
        a_inside = asyncio.Event()
        b_inside = asyncio.Event()
        gate_a = asyncio.Event()
        gate_b = asyncio.Event()

        async def wrapper(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                a_inside.set()
                await gate_a.wait()
            else:
                b_inside.set()
                await gate_b.wait()
            conn = await real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        # Hold the Thread OBJECTS, not their id()s: an id is reusable once the
        # original object is collected, so a Thread freed mid-test whose address
        # a newly-created worker inherits would be silently classified as
        # pre-existing and the survivors assertion below would go quiet.  The
        # strong references are what make the comparison mean anything.  Matches
        # test_durability.py's baseline.
        before = set(live_aiosqlite_worker_threads())
        try:
            with patch('aiosqlite.connect', wrapper):
                # (1) Getter A suspends mid-connect, then is cancelled.  The
                #     shielded connect keeps running; the per-path lock is
                #     released as A unwinds.
                getter_a = asyncio.create_task(pool.get(db_path))
                await asyncio.wait_for(a_inside.wait(), timeout=2.0)
                getter_a.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await getter_a

                # (2) Getter B takes the freed lock and starts its OWN connect
                #     for the same path — A's is still in flight, so B sees an
                #     empty _conns and cannot know one is landing.
                getter_b = asyncio.create_task(pool.get(db_path))
                await asyncio.wait_for(b_inside.wait(), timeout=2.0)

                # (3) A lands first and is ADOPTED into the pool.
                gate_a.set()
                await _wait_until(lambda: pool.open_count == 1)

                # (4) B lands second, resuming into the post-shield re-check.
                gate_b.set()
                result_b = await asyncio.wait_for(getter_b, timeout=2.0)

            assert len(opened) == 2, (
                f'expected 2 physical connects (A adopted, B racing), got '
                f'{len(opened)}'
            )
            assert pool.open_count == 1, (
                f'expected exactly one pooled connection for the path, got '
                f'{pool.open_count}'
            )
            assert pool._conns[db_path.resolve()] is opened[0], (
                'the adopted connection was overwritten in _conns; it is now '
                'in neither _conns nor _inflight, so close_all() can never '
                'reap it and its aiosqlite worker thread is unreachable'
            )
            assert result_b is opened[0], (
                f'expected the racing getter to converge on the pooled '
                f'connection, got {result_b!r} (is opened[0]: '
                f'{result_b is opened[0]}, is opened[1]: '
                f'{result_b is opened[1]})'
            )

            # close_all() must reap BOTH: the adopted one it holds, and the
            # loser B opened (which get() is responsible for closing).
            await pool.close_all()
            assert pool.open_count == 0, (
                f'expected open_count=0 after close_all, got {pool.open_count}'
            )
            # Verified against aiosqlite >=0.22.x — see the private-attribute
            # pin rationale on the close_all race test above.
            for index, conn in enumerate(opened):
                assert (
                    hasattr(conn, '_connection')
                    and hasattr(conn, '_running')
                    and hasattr(conn, '_thread')
                ), 'aiosqlite internal attribute names changed — update test'
                assert conn._connection is None, (
                    f'opened[{index}] not closed (_connection is '
                    f'{conn._connection!r})'
                )
                assert conn._running is False, (
                    f'opened[{index}] worker still running (_running is '
                    f'{conn._running!r})'
                )
                conn._thread.join(timeout=2.0)
                assert not conn._thread.is_alive(), (
                    f'opened[{index}] aiosqlite worker thread did not exit: '
                    f'an adopted connection was overwritten in _conns; its '
                    f'aiosqlite worker thread is unreachable by close_all()'
                )

            survivors = [
                t for t in live_aiosqlite_worker_threads() if t not in before
            ]
            assert not survivors, (
                f'{len(survivors)} aiosqlite worker thread(s) survived '
                f'close_all(): an adopted connection was overwritten in '
                f'_conns; its aiosqlite worker thread is unreachable by '
                f'close_all()'
            )
        finally:
            gate_a.set()
            gate_b.set()
            await _reap(opened)

    async def test_close_all_does_not_return_while_adopted_close_is_in_flight(
        self, tmp_path
    ):
        """close_all() must not RETURN while a close it caused is still running.

        Current behaviour (the thing under test): ``_close_once`` JOINS a close
        already in flight for the same ``Connection``, awaiting the shared
        future under ``asyncio.shield`` instead of returning.  ``close_all()``
        additionally gathers the fire-and-forget close tasks ``_adopt_or_close``
        scheduled, so it cannot return while one is still running.

        The RED condition this pins is history: BEFORE step-12, ``_close_once``
        early-returned on ``conn in self._closing`` — it SKIPPED rather than
        JOINED.  During ``close_all()`` the ordering is deterministic: the
        landing connect fires ``_adopt_or_close`` (whose done-callback was
        registered in ``get()``, hence ahead of ``asyncio.wait``'s own), which
        with ``_closed`` already True schedules a FIRE-AND-FORGET ``_close_once``
        task; that task's first step is queued via ``call_soon`` ahead of
        ``_drain_inflight``'s resumption, so it entered ``await conn.close()``
        first and ``_drain_inflight``'s own ``_close_once(...)`` then hit the
        ``_closing`` guard and returned instantly.  Nothing awaited the
        scheduled task, so ``close_all()`` returned with the close in flight.

        Why that is a real bug and not just untidy: under uvicorn — or a
        ``TestClient`` portal that closes the loop promptly after shutdown — the
        pending close task is destroyed mid-flight and the aiosqlite worker
        thread is stranded.  That is the ORIGINAL task-3466 defect reached by a
        new route.

        The assertions below run with NO intervening sleep, and the ABSENCE of
        that sleep is the whole point: the step-1/step-7 durability guards pass
        today only because they sleep after the lifespan block.  Do not add one.
        """
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        real_connect = aiosqlite.connect
        opened: list[aiosqlite.Connection] = []

        inside_connect = asyncio.Event()
        gate = asyncio.Event()

        async def wrapper(*args, **kwargs):
            inside_connect.set()
            await gate.wait()
            conn = await real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        try:
            with patch('aiosqlite.connect', wrapper):
                getter = asyncio.create_task(pool.get(db_path))
                await asyncio.wait_for(inside_connect.wait(), timeout=2.0)
                getter.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await getter

                closer = asyncio.create_task(pool.close_all())
                # Let close_all() reach _drain_inflight before the connect lands,
                # so the landing happens MID-DRAIN (the deterministic ordering
                # described above).  _INFLIGHT_DRAIN_TIMEOUT is deliberately NOT
                # shrunk: this case needs the connect to land INSIDE the budget,
                # and it lands as soon as `gate` is set.
                await asyncio.sleep(0)
                gate.set()
                await asyncio.wait_for(closer, timeout=5.0)

            assert len(opened) == 1, (
                f'expected exactly 1 physical connect, got {len(opened)}'
            )
            conn = opened[0]
            # NO sleep here — see the docstring.
            assert conn._running is False, (
                'close_all() returned while a connection it owned was still '
                'being closed; if the loop closes now the aiosqlite worker '
                f'thread is stranded (_running is {conn._running!r})'
            )
            assert conn._connection is None, (
                'close_all() returned while a connection it owned was still '
                'being closed; if the loop closes now the aiosqlite worker '
                f'thread is stranded (_connection is {conn._connection!r})'
            )
            # The thread's own exit happens just after the STOP sentinel is
            # processed, so join() the brief post-STOP window before asserting —
            # exactly as the close_all race test above does.
            conn._thread.join(timeout=2.0)
            assert not conn._thread.is_alive(), (
                'aiosqlite worker thread did not exit after close_all returned'
            )
        finally:
            gate.set()
            # _reap() cannot be called blind here.  On a RED run the close this
            # test is about is STILL IN FLIGHT, and a second concurrent close()
            # queues a second STOP sentinel that nothing will ever resolve — the
            # permanent hang _close_once exists to prevent.  Let the in-flight
            # close settle first (it needs no help; it is already running), then
            # reap only what is genuinely still open.
            for conn in opened:
                with contextlib.suppress(TimeoutError):
                    await _wait_until(
                        lambda c=conn: not getattr(c, '_running', False),
                        timeout=2.0,
                    )
            await _reap(opened)

    async def test_close_all_waits_for_straggler_adopted_after_drain_budget(
        self, tmp_path, monkeypatch
    ):
        """close_all() must also wait for a straggler that lands AFTER the budget.

        The residual window the ``_close_once`` join does not cover: a connect
        that misses ``_INFLIGHT_DRAIN_TIMEOUT`` is in ``still_pending``, not
        ``done``, so ``_drain_inflight`` never awaits it.  If it lands while
        ``close_all()`` is still inside its ``_conns`` loop, ``_adopt_or_close``
        schedules a fire-and-forget close that nothing joins — and ``close_all()``
        returns with that close in flight, stranding the worker thread if the
        loop is torn down promptly (uvicorn, or a TestClient portal).

        Choreography — X is a normal pooled connection whose close is gated so
        ``close_all()`` can be held inside its ``_conns`` loop; Y is the
        straggler.  Both closes are gated by replacing the INSTANCE ``close``
        method: verified on the pinned aiosqlite that ``Connection`` has no
        ``__slots__``, so per-instance assignment works and gives deterministic
        control that patching the class would not.

        Deterministic in BOTH directions.  RED: once ``release_x`` is set the
        only work left in ``close_all()`` is a dict clear, so 0.1s is ample for
        ``closer`` to finish.  GREEN: ``closer`` CANNOT finish, because it is
        gathering Y's close task, which is blocked on ``release_y``.

        The step-6 straggler WARNING fires here — Y misses the budget by
        construction.  That is expected; this test does not assert against it.
        """
        monkeypatch.setattr('dashboard.data.db._INFLIGHT_DRAIN_TIMEOUT', 0.01)
        db_x = tmp_path / 'x.db'
        db_y = tmp_path / 'y.db'
        sqlite3.connect(str(db_x)).close()
        sqlite3.connect(str(db_y)).close()

        pool = DbPool()
        real_connect = aiosqlite.connect
        opened: list[aiosqlite.Connection] = []

        x_close_started = asyncio.Event()
        release_x = asyncio.Event()
        y_close_started = asyncio.Event()
        release_y = asyncio.Event()
        inside_y = asyncio.Event()
        gate_y = asyncio.Event()

        # X opens through the UNPATCHED connect, so _conns is non-empty and
        # close_all() has something to iterate.
        conn_x = await pool.get(db_x)
        assert conn_x is not None
        opened.append(conn_x)
        real_close_x = conn_x.close

        async def gated_close_x():
            x_close_started.set()
            await release_x.wait()
            await real_close_x()

        conn_x.close = gated_close_x

        async def wrapper(*args, **kwargs):
            inside_y.set()
            await gate_y.wait()
            conn = await real_connect(*args, **kwargs)
            real_close_y = conn.close

            async def gated_close_y():
                y_close_started.set()
                await release_y.wait()
                await real_close_y()

            conn.close = gated_close_y
            opened.append(conn)
            return conn

        try:
            with patch('aiosqlite.connect', wrapper):
                getter_y = asyncio.create_task(pool.get(db_y))
                await asyncio.wait_for(inside_y.wait(), timeout=2.0)
                getter_y.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await getter_y

                closer = asyncio.create_task(pool.close_all())
                # close_all() has now blown the 0.01s drain budget (Y is a
                # logged straggler) and is suspended inside the _conns loop.
                await asyncio.wait_for(x_close_started.wait(), timeout=2.0)

                # Y lands late; _adopt_or_close sees _closed=True and schedules
                # its close, which then enters conn.close().
                gate_y.set()
                await asyncio.wait_for(y_close_started.wait(), timeout=2.0)

                # Let X's close finish.  Everything close_all() itself still has
                # to do is now synchronous.
                release_x.set()
                await asyncio.sleep(0.1)

                assert not closer.done(), (
                    'close_all() returned while a straggler connection adopted '
                    'during shutdown was still being closed; if the loop is torn '
                    'down now its aiosqlite worker thread is stranded'
                )

                release_y.set()
                await asyncio.wait_for(closer, timeout=2.0)

            assert len(opened) == 2, (
                f'expected 2 connections (pooled X, straggler Y), got '
                f'{len(opened)}'
            )
            conn_y = opened[1]
            assert conn_y._running is False, (
                f'straggler Y not closed (_running is {conn_y._running!r})'
            )
            assert conn_y._connection is None, (
                f'straggler Y not closed (_connection is {conn_y._connection!r})'
            )
            conn_y._thread.join(timeout=2.0)
            assert not conn_y._thread.is_alive(), (
                'the straggler aiosqlite worker thread did not exit after '
                'close_all returned'
            )
        finally:
            gate_y.set()
            release_x.set()
            release_y.set()
            # As in the in-flight-close test above: never reap blind while a
            # close may still be running — a second concurrent close() queues a
            # STOP sentinel nothing will resolve.
            for conn in opened:
                with contextlib.suppress(TimeoutError):
                    await _wait_until(
                        lambda c=conn: not getattr(c, '_running', False),
                        timeout=2.0,
                    )
            await _reap(opened)

    async def test_close_all_waits_for_straggler_adopted_during_the_final_gather(
        self, tmp_path, monkeypatch
    ):
        """close_all() must join a close scheduled AFTER its join already started.

        The last residual window, and the reason ``close_all()`` drains
        ``_pending_closes`` to a FIXED POINT rather than gathering one snapshot.
        A straggler that missed ``_INFLIGHT_DRAIN_TIMEOUT`` is still in
        ``_inflight`` with ``_adopt_or_close`` attached, so it can land *while*
        that gather is awaiting: the close it schedules is added to
        ``_pending_closes`` after the snapshot was taken and a single-shot
        gather would never join it.  ``close_all()`` would then return with a
        close in flight — the same stranded-worker-thread state as the two
        tests above, reached by a third route, and a direct contradiction of the
        invariant :meth:`DbPool.close_all`'s docstring states.

        Choreography — three connections, all gated on their INSTANCE ``close``
        as in the test above.  X is pooled, so ``close_all()`` can be held inside
        its ``_conns`` loop.  Y is a straggler that lands BEFORE the gather, so
        it is in the snapshot.  Z is a straggler that lands DURING it.

        The barrier between "Y is in the snapshot" and "Z lands during the
        gather" is exact, not a sleep: ``pool.open_count`` drops to 0 at
        ``_conns.clear()``, and between that and the gather ``close_all()``
        executes no ``await``, so it cannot be observed anywhere in between.
        Once the poll sees 0, ``close_all()`` is necessarily parked in the
        gather.

        Deterministic in BOTH directions.  RED: once ``release_y`` completes the
        snapshot's only task, nothing remains but a ``return``, so 0.1s is ample
        for ``closer`` to finish.  GREEN: ``closer`` CANNOT finish, because the
        re-check finds Z's close and gathers it, and that one is blocked on
        ``release_z``.

        The straggler WARNING fires here — Y and Z both miss the budget by
        construction.  That is expected; this test does not assert against it.
        """
        monkeypatch.setattr('dashboard.data.db._INFLIGHT_DRAIN_TIMEOUT', 0.01)
        db_x = tmp_path / 'x.db'
        db_y = tmp_path / 'y.db'
        db_z = tmp_path / 'z.db'
        for path in (db_x, db_y, db_z):
            sqlite3.connect(str(path)).close()

        pool = DbPool()
        real_connect = aiosqlite.connect
        opened: list[aiosqlite.Connection] = []

        x_close_started = asyncio.Event()
        release_x = asyncio.Event()
        started: dict[str, asyncio.Event] = {'y': asyncio.Event(), 'z': asyncio.Event()}
        release: dict[str, asyncio.Event] = {'y': asyncio.Event(), 'z': asyncio.Event()}
        inside: dict[str, asyncio.Event] = {'y': asyncio.Event(), 'z': asyncio.Event()}
        gate: dict[str, asyncio.Event] = {'y': asyncio.Event(), 'z': asyncio.Event()}

        conn_x = await pool.get(db_x)
        assert conn_x is not None
        opened.append(conn_x)
        real_close_x = conn_x.close

        async def gated_close_x():
            x_close_started.set()
            await release_x.wait()
            await real_close_x()

        conn_x.close = gated_close_x

        async def wrapper(*args, **kwargs):
            # Keyed on the path rather than a call counter: the two stragglers
            # open DIFFERENT databases, so this cannot mis-attribute a gate.
            key = 'z' if 'z.db' in str(args[0]) else 'y'
            inside[key].set()
            await gate[key].wait()
            conn = await real_connect(*args, **kwargs)
            real_close = conn.close

            async def gated_close(k=key, rc=real_close):
                started[k].set()
                await release[k].wait()
                await rc()

            conn.close = gated_close
            opened.append(conn)
            return conn

        async def cancelled_getter(db_path, key):
            getter = asyncio.create_task(pool.get(db_path))
            await asyncio.wait_for(inside[key].wait(), timeout=2.0)
            getter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await getter

        try:
            with patch('aiosqlite.connect', wrapper):
                await cancelled_getter(db_y, 'y')
                await cancelled_getter(db_z, 'z')

                closer = asyncio.create_task(pool.close_all())
                # Both stragglers have now blown the 0.01s drain budget and
                # close_all() is suspended inside the _conns loop on X.
                await asyncio.wait_for(x_close_started.wait(), timeout=2.0)

                # Y lands first, so its close IS in the snapshot the gather
                # would take.
                gate['y'].set()
                await asyncio.wait_for(started['y'].wait(), timeout=2.0)

                # Exact barrier: open_count hits 0 at _conns.clear(), and
                # close_all() does not await again before the gather.
                release_x.set()
                await _wait_until(lambda: pool.open_count == 0)

                # Z lands DURING the gather — after the snapshot existed.
                gate['z'].set()
                await asyncio.wait_for(started['z'].wait(), timeout=2.0)

                # Retire the snapshot's only member.  A single-shot gather now
                # has nothing left to wait on.
                release['y'].set()
                await asyncio.sleep(0.1)

                assert not closer.done(), (
                    'close_all() returned while a straggler adopted DURING its '
                    'final join was still being closed; a single snapshot of '
                    '_pending_closes cannot see a close scheduled after it was '
                    'taken, so the worker thread is stranded if the loop is '
                    'torn down now'
                )

                release['z'].set()
                await asyncio.wait_for(closer, timeout=2.0)

            assert len(opened) == 3, (
                f'expected 3 connections (pooled X, stragglers Y and Z), got '
                f'{len(opened)}'
            )
            conn_z = opened[2]
            assert conn_z._running is False, (
                f'straggler Z not closed (_running is {conn_z._running!r})'
            )
            assert conn_z._connection is None, (
                f'straggler Z not closed (_connection is {conn_z._connection!r})'
            )
            conn_z._thread.join(timeout=2.0)
            assert not conn_z._thread.is_alive(), (
                'the aiosqlite worker thread of the straggler adopted during '
                'the final gather did not exit after close_all returned'
            )
        finally:
            gate['y'].set()
            gate['z'].set()
            release_x.set()
            release['y'].set()
            release['z'].set()
            for conn in opened:
                with contextlib.suppress(TimeoutError):
                    await _wait_until(
                        lambda c=conn: not getattr(c, '_running', False),
                        timeout=2.0,
                    )
            await _reap(opened)

    async def test_close_all_warns_when_inflight_connect_exceeds_drain_budget(
        self, tmp_path, caplog, monkeypatch
    ):
        """A connect that outlives the drain budget must be reported, not dropped.

        Two invariants in one, both load-bearing:

        1. ``close_all()`` RETURNS.  The drain budget is bounded on purpose — an
           unbounded wait would let one wedged sqlite open hang application
           shutdown forever (the failure mode ``_probe_db`` exists to avoid) and
           would turn a leak into a hang inside pytest teardown.
        2. The residual is announced LOUDLY, with the concrete db path.  A
           bounded wait ALONE is a silent fail-soft: the pool would return
           knowing an aiosqlite worker thread may outlive the event loop, and
           that fact is unrecoverable anywhere else (INV-2,
           structured-facts-at-failure / no-silent-fail-soft).
        """
        monkeypatch.setattr('dashboard.data.db._INFLIGHT_DRAIN_TIMEOUT', 0.05)
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()

        pool = DbPool()
        real_connect = aiosqlite.connect
        opened: list[aiosqlite.Connection] = []

        inside_connect = asyncio.Event()
        # Deliberately NOT set until the assertions are done: the connect can
        # therefore never land inside the drain budget.
        release = asyncio.Event()

        async def wrapper(*args, **kwargs):
            inside_connect.set()
            await release.wait()
            conn = await real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        try:
            with patch('aiosqlite.connect', wrapper):
                getter = asyncio.create_task(pool.get(db_path))
                await asyncio.wait_for(inside_connect.wait(), timeout=2.0)
                getter.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await getter

                caplog.clear()
                with caplog.at_level(logging.WARNING, logger='dashboard.data.db'):
                    # (1) Bounded: this must not hang on the stuck connect.
                    await asyncio.wait_for(pool.close_all(), timeout=2.0)

                # (2) Loud: exactly one WARNING, naming the path and the budget.
                warnings = [
                    r
                    for r in caplog.records
                    if r.name == 'dashboard.data.db' and r.levelno == logging.WARNING
                ]
                assert len(warnings) == 1, (
                    f'expected exactly 1 WARNING about the undrained connect, '
                    f'got {len(warnings)}: {[r.getMessage() for r in warnings]}'
                )
                message = warnings[0].getMessage()
                assert str(db_path.resolve()) in message, (
                    f'straggler WARNING must name the still-pending db path '
                    f'{db_path.resolve()}; got: {message!r}'
                )
                assert 'drain' in message.lower() or 'land' in message.lower(), (
                    f'straggler WARNING must state that the connect did not '
                    f'land within the drain budget; got: {message!r}'
                )

                # Release the stuck connect and let the pool reap the late
                # arrival, so this test leaves no live worker thread behind.
                release.set()
                await _wait_until(lambda: bool(opened))
                await _wait_until(lambda: not opened[0]._running)
        finally:
            release.set()
            await _reap(opened)


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
