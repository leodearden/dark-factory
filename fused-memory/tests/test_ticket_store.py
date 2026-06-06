"""Tests for the TicketStore SQLite persistence layer (two-phase add_task)."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from test_daemon_connect_consolidation import assert_connection_thread_is_daemon

from fused_memory.middleware.ticket_store import TicketStore, _new_ticket_id


@pytest_asyncio.fixture
async def store(tmp_path):
    s = TicketStore(tmp_path / 'tickets.db')
    await s.initialize()
    yield s
    await s.close()


def _assert_connection_closed(conn) -> None:
    """Assert that an aiosqlite Connection is fully closed.

    Checks ``Connection._connection`` (set to ``None`` by ``close()``).
    Isolated here so the internal-attribute dependency lives in one place — a
    future aiosqlite rename only needs updating here (task 1560).
    """
    assert conn._connection is None, (
        'aiosqlite connection was not closed — its non-daemon worker thread '
        'stays alive and raises "Event loop is closed" on GC (task 1560)'
    )


@pytest.mark.asyncio
async def test_initialize_creates_schema_and_reinit_after_close_is_safe(tmp_path):
    """initialize() creates the tickets table; re-initializing after close() does not raise.

    Tests the init→close→init cycle (safe usage pattern).  Back-to-back
    initialize() without an intervening close() is not covered here because
    initialize() unconditionally opens a new connection, orphaning the previous
    one.  The production footgun is out of scope for task 1560 (flagged via
    escalate_info); tests exercise the safe pattern and guard against leaks.
    """
    store = TicketStore(tmp_path / 'tickets.db')
    await store.initialize()
    # task 1560: capture the first connection before the idempotent re-init;
    # initialize() unconditionally opens a new aiosqlite connection, orphaning
    # the previous one if not explicitly closed first.  The orphaned non-daemon
    # worker thread raises "Event loop is closed" on GC.
    first_db = store._db
    # Close the first connection before the idempotent re-init so it is not
    # orphaned (task 1560: its non-daemon worker would otherwise leak).
    await store.close()
    # Second call must be idempotent (CREATE IF NOT EXISTS)
    await store.initialize()
    try:
        # Verify the table exists with the expected columns
        db = store._db
        assert db is not None
        cursor = await db.execute("PRAGMA table_info(tickets)")
        rows = await cursor.fetchall()
        col_names = {row[1] for row in rows}

        expected_columns = {
            'ticket_id',
            'project_id',
            'candidate_json',
            'status',
            'task_id',
            'reason',
            'result_json',
            'created_at',
            'resolved_at',
            'expires_at',
            'escalated_at',
        }
        assert expected_columns == col_names, (
            f"Missing columns: {expected_columns - col_names}; "
            f"Extra columns: {col_names - expected_columns}"
        )
    finally:
        await store.close()
    # task 1560: every connection this test opened must now be closed.
    assert first_db is not None, 'first_db should have been set after initialize()'
    _assert_connection_closed(first_db)
    assert store._db is None, 'store._db must be None after close() (task 1560)'


@pytest.mark.asyncio
async def test_double_initialize_without_close_is_idempotent_and_no_leak(tmp_path):
    """A second initialize() WITHOUT an intervening close() is safe and leaks no connection.

    Covers the back-to-back init→init path (task 1562).  After the second
    initialize():
    - The prior connection is CLOSED (not orphaned).
    - The new connection is a fresh, daemon-backed worker.
    - The store is fully usable through the new connection.

    This test is RED on current code because initialize() unconditionally opens a
    new connection, orphaning the prior one (task 1562).
    """
    store = TicketStore(tmp_path / 'tickets.db')
    await store.initialize()
    first_db = store._db
    assert first_db is not None

    # Second initialize() WITHOUT an intervening close() — the idempotency path.
    store_second_db = None
    try:
        await store.initialize()
        store_second_db = store._db

        # (a) A fresh connection was opened.
        assert store._db is not None and store._db is not first_db, (
            'store._db should be a new connection after the second initialize()'
        )

        # (b) RED ASSERTION: the prior connection must be closed, not orphaned.
        # On current code first_db._connection is still set → AssertionError.
        # Guarantees no "Event loop is closed" ResourceWarning on GC because
        # aiosqlite.Connection.__del__ early-returns when _connection is None.
        # NOTE: _assert_connection_closed() is the deterministic contract here —
        # aiosqlite sets ._connection = None synchronously in close()'s finally.
        # The worker thread exits asynchronously after the stop-future resolves,
        # so asserting thread.is_alive() races teardown and can flake on loaded
        # CI.  Thread death is best-effort; do not poll or assert it here
        # (amendment pass, suggestion 1).
        _assert_connection_closed(first_db)

        # (c) The NEW connection is a live daemon-backed worker.
        assert_connection_thread_is_daemon(store._db, 'TicketStore re-init')

        # (d) Usability: submit + get through the new connection.
        tid = await store.submit(project_id='p', candidate_json='{}')
        row = await store.get(tid)
        assert row is not None and row['status'] == 'pending'

    finally:
        await store.close()
        assert store._db is None, 'store._db must be None after close()'
        if store_second_db is not None:
            _assert_connection_closed(store_second_db)


@pytest.mark.asyncio
async def test_reconnect_close_then_initialize_preserves_data_and_no_leak(tmp_path):
    """Simulated reconnect (close→initialize) preserves data and opens a fresh connection.

    Net-new coverage over test_initialize_creates_schema_and_reinit_after_close_is_safe:
    data persistence across the close()→initialize() cycle.  Schema idempotency and
    the connection-closed / no-leak guarantees on the explicit-close path are already
    covered by that test; this test pins the data-persistence contract against
    regression (amendment pass, suggestion 2).
    """
    store = TicketStore(tmp_path / 'tickets.db')
    await store.initialize()
    tid = await store.submit(project_id='p', candidate_json='{}')
    first_db = store._db
    assert first_db is not None

    # Explicit close — the canonical safe teardown path.
    await store.close()
    assert store._db is None

    # Reconnect via initialize().
    try:
        await store.initialize()

        # New connection is a fresh daemon-backed worker, distinct from the prior one.
        assert store._db is not None and store._db is not first_db
        assert_connection_thread_is_daemon(store._db, 'TicketStore reconnect')

        # Data survived the reconnect — net-new assertion over the schema-only test.
        row = await store.get(tid)
        assert row is not None and row['status'] == 'pending', (
            f'ticket {tid!r} not found after reconnect'
        )

    finally:
        await store.close()
        assert store._db is None


@pytest.mark.asyncio
async def test_new_ticket_id_has_tkt_prefix_and_sorts_by_time():
    """_new_ticket_id() returns tkt_-prefixed ids that are lexicographically time-ordered."""
    id1 = _new_ticket_id()
    await asyncio.sleep(0.001)  # ensure nanosecond timestamp advances
    id2 = _new_ticket_id()

    # Both must start with tkt_
    assert id1.startswith('tkt_'), f"id1 missing tkt_ prefix: {id1!r}"
    assert id2.startswith('tkt_'), f"id2 missing tkt_ prefix: {id2!r}"

    # Exactly the documented length: 4 (prefix) + 26 (crockford base32 of 16 bytes)
    assert len(id1) == 30, f"Expected length 30, got {len(id1)}"
    assert len(id2) == 30, f"Expected length 30, got {len(id2)}"

    # Later id sorts after the earlier one (monotonic time-ordered)
    assert id1 < id2, f"Expected id1 < id2 but got {id1!r} >= {id2!r}"


@pytest.mark.asyncio
async def test_submit_persists_pending_ticket_and_returns_id(store):
    """submit() inserts a pending row and returns a tkt_-prefixed id."""
    candidate = json.dumps({'title': 'Test Task', 'description': 'Do it'})
    ticket_id = await store.submit(project_id='p', candidate_json=candidate)

    assert ticket_id.startswith('tkt_')

    # Verify the persisted row
    row = await store.get(ticket_id)
    assert row is not None
    assert row['status'] == 'pending'
    assert row['project_id'] == 'p'
    assert row['candidate_json'] == candidate

    # created_at must be set; expires_at is now an advisory placeholder set
    # far in the future (worker-liveness reaper supplanted wall-clock TTL).
    created_at = datetime.fromisoformat(row['created_at'])
    expires_at = datetime.fromisoformat(row['expires_at'])
    assert created_at.tzinfo is not None  # timezone-aware
    assert expires_at > created_at + timedelta(days=180), (
        f'expires_at must be far-future placeholder, got {expires_at}'
    )

    # Unresolved columns must be NULL
    assert row['task_id'] is None
    assert row['reason'] is None
    assert row['resolved_at'] is None
    assert row['result_json'] is None


@pytest.mark.asyncio
async def test_get_returns_row_or_none(store):
    """get() returns a dict for known tickets and None for unknown ids."""
    candidate = json.dumps({'title': 'T'})
    ticket_id = await store.submit(project_id='proj', candidate_json=candidate)

    row = await store.get(ticket_id)
    assert row is not None
    assert isinstance(row, dict)
    expected_keys = {
        'ticket_id', 'project_id', 'candidate_json', 'status',
        'task_id', 'reason', 'result_json', 'created_at', 'resolved_at',
        'expires_at', 'escalated_at',
    }
    assert expected_keys == set(row.keys())
    assert row['ticket_id'] == ticket_id

    missing = await store.get('tkt_nonexistent_000000000000')
    assert missing is None


@pytest.mark.asyncio
async def test_mark_resolved_sets_terminal_status_and_resolved_at(store):
    """mark_resolved() sets status, resolved_at, task_id, reason, result_json."""
    candidate = json.dumps({'title': 'T'})

    # --- created ---
    tid = await store.submit(project_id='p', candidate_json=candidate)
    result = await store.mark_resolved(tid, status='created', task_id='42', result_json='{"id":"42"}')
    assert result is True
    row = await store.get(tid)
    assert row['status'] == 'created'
    assert row['task_id'] == '42'
    assert row['result_json'] == '{"id":"42"}'
    assert row['resolved_at'] is not None

    # --- combined (task_id populated, reason optional) ---
    tid2 = await store.submit(project_id='p', candidate_json=candidate)
    await store.mark_resolved(tid2, status='combined', task_id='5', reason='dedup')
    row2 = await store.get(tid2)
    assert row2['status'] == 'combined'
    assert row2['task_id'] == '5'
    assert row2['reason'] == 'dedup'

    # --- dropped (task_id None, reason populated) ---
    tid3 = await store.submit(project_id='p', candidate_json=candidate)
    await store.mark_resolved(tid3, status='dropped', reason='backlog_full')
    row3 = await store.get(tid3)
    assert row3['status'] == 'dropped'
    assert row3['task_id'] is None
    assert row3['reason'] == 'backlog_full'

    # --- failed ---
    tid4 = await store.submit(project_id='p', candidate_json=candidate)
    await store.mark_resolved(tid4, status='failed', reason='db locked')
    row4 = await store.get(tid4)
    assert row4['status'] == 'failed'
    assert row4['reason'] == 'db locked'
    assert row4['resolved_at'] is not None

    # Double-resolve attempt: mark_resolved on an already-resolved ticket
    # returns False (no-op) without clobbering the existing data.
    result_again = await store.mark_resolved(tid, status='failed', reason='clobber attempt')
    assert result_again is False
    row_after = await store.get(tid)
    assert row_after['status'] == 'created'  # unchanged


@pytest.mark.asyncio
async def test_flush_pending_on_startup_marks_all_pending_failed(store):
    """flush_pending_on_startup() marks all pending tickets failed/server_restart."""
    candidate = json.dumps({'title': 'T'})

    # Submit three pending tickets
    t1 = await store.submit(project_id='p', candidate_json=candidate)
    t2 = await store.submit(project_id='p', candidate_json=candidate)
    t3 = await store.submit(project_id='p', candidate_json=candidate)

    # Resolve one of them before the flush
    await store.mark_resolved(t1, status='created', task_id='99')

    count = await store.flush_pending_on_startup()
    assert count == 2  # two pending tickets flushed

    row2 = await store.get(t2)
    assert row2['status'] == 'failed'
    assert row2['reason'] == 'server_restart'
    assert row2['resolved_at'] is not None

    row3 = await store.get(t3)
    assert row3['status'] == 'failed'
    assert row3['reason'] == 'server_restart'

    # The already-resolved ticket must be untouched
    row1 = await store.get(t1)
    assert row1['status'] == 'created'


# ---------------------------------------------------------------------------
# Janitor-facing helpers: fetch_unescalated_failures + mark_escalated
# ---------------------------------------------------------------------------


async def _force_failed(store: TicketStore, ticket_id: str, *, reason: str) -> None:
    """Test helper: terminalise a ticket as failed with the given reason."""
    db = store._db
    assert db is not None
    now = datetime.now(UTC).isoformat()
    await db.execute(
        "UPDATE tickets SET status = 'failed', reason = ?, resolved_at = ? "
        "WHERE ticket_id = ?",
        (reason, now, ticket_id),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_fetch_unescalated_failures_returns_only_failed_null_escalated(store):
    """fetch_unescalated_failures excludes pending, combined, idempotency_hit,
    and rows already stamped via mark_escalated."""
    pending_id = await store.submit(project_id='p', candidate_json='{}')
    failed_id = await store.submit(project_id='p', candidate_json='{}')
    await _force_failed(store, failed_id, reason='curator_failed')
    idem_id = await store.submit(project_id='p', candidate_json='{}')
    await _force_failed(store, idem_id, reason='idempotency_hit')

    rows = await store.fetch_unescalated_failures()
    ids = {r['ticket_id'] for r in rows}
    assert failed_id in ids
    assert pending_id not in ids
    assert idem_id not in ids, (
        'idempotency_hit must be excluded — happy-path even when status=failed'
    )

    # After mark_escalated, the row no longer surfaces.
    n = await store.mark_escalated([failed_id])
    assert n == 1
    rows_after = await store.fetch_unescalated_failures()
    assert all(r['ticket_id'] != failed_id for r in rows_after)


@pytest.mark.asyncio
async def test_fetch_unescalated_failures_filters_by_project(store):
    a_id = await store.submit(project_id='proj-a', candidate_json='{}')
    b_id = await store.submit(project_id='proj-b', candidate_json='{}')
    await _force_failed(store, a_id, reason='curator_failed')
    await _force_failed(store, b_id, reason='curator_failed')

    a_only = await store.fetch_unescalated_failures(project_id='proj-a')
    assert {r['ticket_id'] for r in a_only} == {a_id}

    both = await store.fetch_unescalated_failures()
    assert {r['ticket_id'] for r in both} == {a_id, b_id}


@pytest.mark.asyncio
async def test_fetch_unescalated_failures_orders_by_resolved_at(store):
    older = await store.submit(project_id='p', candidate_json='{}')
    newer = await store.submit(project_id='p', candidate_json='{}')
    # Force resolved_at directly so ordering is deterministic in CI.
    db = store._db
    await db.execute(
        "UPDATE tickets SET status='failed', reason='r', resolved_at=? WHERE ticket_id=?",
        ('2026-01-01T00:00:00+00:00', older),
    )
    await db.execute(
        "UPDATE tickets SET status='failed', reason='r', resolved_at=? WHERE ticket_id=?",
        ('2026-02-01T00:00:00+00:00', newer),
    )
    await db.commit()

    rows = await store.fetch_unescalated_failures()
    assert [r['ticket_id'] for r in rows] == [older, newer]


@pytest.mark.asyncio
async def test_mark_escalated_bulk_updates(store):
    a_id = await store.submit(project_id='p', candidate_json='{}')
    b_id = await store.submit(project_id='p', candidate_json='{}')
    c_id = await store.submit(project_id='p', candidate_json='{}')
    await _force_failed(store, a_id, reason='r')
    await _force_failed(store, b_id, reason='r')
    await _force_failed(store, c_id, reason='r')

    n = await store.mark_escalated([a_id, b_id])
    assert n == 2
    a = await store.get(a_id)
    b = await store.get(b_id)
    c = await store.get(c_id)
    assert a['escalated_at'] is not None
    assert b['escalated_at'] is not None
    assert c['escalated_at'] is None


@pytest.mark.asyncio
async def test_mark_escalated_empty_is_noop(store):
    n = await store.mark_escalated([])
    assert n == 0


@pytest.mark.asyncio
async def test_migration_adds_escalated_at_to_legacy_db(tmp_path):
    """An existing DB without the escalated_at column gets the column added."""
    import aiosqlite

    db_path = tmp_path / 'tickets.db'
    # Create a legacy schema (no escalated_at).
    legacy_conn = await aiosqlite.connect(str(db_path))
    try:
        await legacy_conn.execute("""
            CREATE TABLE tickets (
                ticket_id   TEXT PRIMARY KEY,
                project_id  TEXT NOT NULL,
                candidate_json TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                task_id     TEXT,
                reason      TEXT,
                result_json TEXT,
                created_at  TEXT NOT NULL,
                resolved_at TEXT,
                expires_at  TEXT NOT NULL
            )
        """)
        await legacy_conn.commit()
    finally:
        await legacy_conn.close()

    # Initialise — the migration must add escalated_at.
    store = TicketStore(db_path)
    await store.initialize()
    # task 1560: capture the first connection before the idempotent re-init;
    # initialize() unconditionally opens a new aiosqlite connection, orphaning
    # the previous one if not explicitly closed first.  The orphaned non-daemon
    # worker thread raises "Event loop is closed" on GC.
    first_db = store._db
    try:
        assert store._db is not None
        cursor = await store._db.execute('PRAGMA table_info(tickets)')
        cols = {r[1] for r in await cursor.fetchall()}
        assert 'escalated_at' in cols
        # Close the first connection before the idempotent re-init so it is not
        # orphaned (task 1560).
        await store.close()
        # And idempotent: re-initialising must not raise.
        await store.initialize()
    finally:
        await store.close()
    # task 1560: every connection this test opened must now be closed.
    assert first_db is not None, 'first_db should have been set after initialize()'
    _assert_connection_closed(first_db)
    assert store._db is None, 'store._db must be None after close() (task 1560)'


# ---------------------------------------------------------------------------
# list_tickets — caller-facing discovery
# ---------------------------------------------------------------------------


async def _force_created_at(store: TicketStore, ticket_id: str, when: datetime) -> None:
    """Test helper: rewrite a ticket's created_at so window filters can be exercised."""
    db = store._db
    assert db is not None
    await db.execute(
        'UPDATE tickets SET created_at = ? WHERE ticket_id = ?',
        (when.isoformat(), ticket_id),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_list_tickets_default_window_7d(store):
    """Default window (no since=) returns rows newer than now-7d only."""
    now = datetime.now(UTC)
    recent = await store.submit(project_id='p', candidate_json='{}')
    old = await store.submit(project_id='p', candidate_json='{}')
    await _force_created_at(store, recent, now - timedelta(days=1))
    await _force_created_at(store, old, now - timedelta(days=8))

    rows = await store.list_tickets('p')
    ids = [r['ticket_id'] for r in rows]
    assert recent in ids
    assert old not in ids


@pytest.mark.asyncio
async def test_list_tickets_status_filter_excludes_other_states(store):
    """status='failed' returns only the failed row."""
    pending = await store.submit(project_id='p', candidate_json='{}')
    failed = await store.submit(project_id='p', candidate_json='{}')
    combined = await store.submit(project_id='p', candidate_json='{}')
    await _force_failed(store, failed, reason='curator_failed')
    db = store._db
    await db.execute(
        "UPDATE tickets SET status='combined', resolved_at=? WHERE ticket_id=?",
        (datetime.now(UTC).isoformat(), combined),
    )
    await db.commit()

    rows = await store.list_tickets('p', status='failed')
    ids = {r['ticket_id'] for r in rows}
    assert ids == {failed}, f'expected only {failed}, got {ids}'

    pending_rows = await store.list_tickets('p', status='pending')
    assert {r['ticket_id'] for r in pending_rows} == {pending}


@pytest.mark.asyncio
async def test_list_tickets_orders_newest_first_and_caps_at_limit(store):
    """Rows come back in DESC created_at order and the limit is honoured."""
    now = datetime.now(UTC)
    ids: list[str] = []
    for i in range(5):
        tid = await store.submit(project_id='p', candidate_json='{}')
        await _force_created_at(store, tid, now - timedelta(minutes=i))
        ids.append(tid)
    # ids[0] is newest (offset 0min), ids[4] is oldest (offset 4min).

    rows = await store.list_tickets('p')
    assert [r['ticket_id'] for r in rows] == ids, 'expected newest-first ordering'

    capped = await store.list_tickets('p', limit=2)
    assert [r['ticket_id'] for r in capped] == ids[:2]


@pytest.mark.asyncio
async def test_list_tickets_filters_by_project(store):
    """Rows from another project never leak into the result."""
    a_id = await store.submit(project_id='proj-a', candidate_json='{}')
    b_id = await store.submit(project_id='proj-b', candidate_json='{}')

    try:
        a_rows = await store.list_tickets('proj-a')
        assert {r['ticket_id'] for r in a_rows} == {a_id}
        b_rows = await store.list_tickets('proj-b')
        assert {r['ticket_id'] for r in b_rows} == {b_id}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_mark_pending_failed_for_project_returns_reaped_ids(store):
    """mark_pending_failed_for_project returns the list of reaped ticket_ids.

    Verifies:
    * Return value is exactly the pending ticket_ids in project A (not an int).
    * Reaped rows are now status='failed'/reason='worker_dead' with resolved_at set.
    * A ticket in project B (different project) is NOT reaped.
    * An already-terminal ticket in A (forced to failed) is excluded from the result.
    """
    # Pending tickets for project A.
    a1 = await store.submit(project_id='proj-a', candidate_json='{}')
    a2 = await store.submit(project_id='proj-a', candidate_json='{}')
    a3 = await store.submit(project_id='proj-a', candidate_json='{}')

    # Already-terminal ticket in A — must NOT appear in the return value.
    a_already_failed = await store.submit(project_id='proj-a', candidate_json='{}')
    await _force_failed(store, a_already_failed, reason='curator_failed')

    # Pending ticket in project B — must NOT be reaped.
    b1 = await store.submit(project_id='proj-b', candidate_json='{}')

    # Call under test.
    reaped = await store.mark_pending_failed_for_project('proj-a', reason='worker_dead')

    # Return type must be a list of strings, not an int.
    assert isinstance(reaped, list), f'expected list, got {type(reaped).__name__}'
    assert set(reaped) == {a1, a2, a3}, (
        f'expected exactly the 3 pending A ticket_ids, got {reaped}'
    )

    # Reaped rows are terminal with correct fields.
    for tid in (a1, a2, a3):
        row = await store.get(tid)
        assert row['status'] == 'failed', f'{tid}: {row}'
        assert row['reason'] == 'worker_dead', f'{tid}: {row["reason"]!r}'
        assert row['resolved_at'] is not None, f'{tid}: resolved_at not set'

    # Already-terminal row in A is untouched (reason remains curator_failed).
    row_af = await store.get(a_already_failed)
    assert row_af['reason'] == 'curator_failed'

    # Project B ticket stays pending.
    row_b = await store.get(b1)
    assert row_b['status'] == 'pending', f'B ticket was unexpectedly reaped: {row_b}'
