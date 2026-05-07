"""Tests for the TicketStore SQLite persistence layer (two-phase add_task)."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from fused_memory.middleware.ticket_store import TicketStore, _new_ticket_id


@pytest_asyncio.fixture
async def store(tmp_path):
    s = TicketStore(tmp_path / 'tickets.db')
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_initialize_creates_schema_and_is_idempotent(tmp_path):
    """initialize() creates the tickets table; calling it twice does not raise."""
    store = TicketStore(tmp_path / 'tickets.db')
    await store.initialize()
    # Second call must be idempotent (CREATE IF NOT EXISTS)
    await store.initialize()

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
    await store.close()


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
    ticket_id = await store.submit(project_id='p', candidate_json=candidate, ttl_seconds=600)

    assert ticket_id.startswith('tkt_')

    # Verify the persisted row
    row = await store.get(ticket_id)
    assert row is not None
    assert row['status'] == 'pending'
    assert row['project_id'] == 'p'
    assert row['candidate_json'] == candidate

    # created_at and expires_at must be set
    created_at = datetime.fromisoformat(row['created_at'])
    expires_at = datetime.fromisoformat(row['expires_at'])
    assert created_at.tzinfo is not None  # timezone-aware
    assert expires_at == created_at + timedelta(seconds=600)

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


@pytest.mark.asyncio
async def test_sweep_expired_marks_only_expired_pending_failed(store):
    """sweep_expired() marks pending expired tickets failed; leaves non-expired alone."""
    now = datetime.now(UTC)
    past = now - timedelta(seconds=1)

    # Insert an already-expired ticket (expires_at in the past)
    expired_id = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
    # Manually backdate the expires_at so it appears expired
    db = store._db
    await db.execute(
        'UPDATE tickets SET expires_at = ? WHERE ticket_id = ?',
        (past.isoformat(), expired_id),
    )
    await db.commit()

    # Insert a ticket that won't expire yet
    live_id = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)

    count = await store.sweep_expired(now=now)
    assert count == 1

    expired_row = await store.get(expired_id)
    assert expired_row['status'] == 'failed'
    assert expired_row['reason'] == 'expired'
    assert expired_row['resolved_at'] is not None

    live_row = await store.get(live_id)
    assert live_row['status'] == 'pending'  # untouched


# ---------------------------------------------------------------------------
# Janitor-facing helpers: fetch_unescalated_failures + mark_escalated
# ---------------------------------------------------------------------------


async def _force_failed(store: TicketStore, ticket_id: str, *, reason: str) -> None:
    """Test helper: terminalise a ticket as failed with the given reason."""
    db = store._db
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
    pending_id = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
    failed_id = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
    await _force_failed(store, failed_id, reason='curator_failed')
    idem_id = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
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
    a_id = await store.submit(project_id='proj-a', candidate_json='{}', ttl_seconds=600)
    b_id = await store.submit(project_id='proj-b', candidate_json='{}', ttl_seconds=600)
    await _force_failed(store, a_id, reason='curator_failed')
    await _force_failed(store, b_id, reason='curator_failed')

    a_only = await store.fetch_unescalated_failures(project_id='proj-a')
    assert {r['ticket_id'] for r in a_only} == {a_id}

    both = await store.fetch_unescalated_failures()
    assert {r['ticket_id'] for r in both} == {a_id, b_id}


@pytest.mark.asyncio
async def test_fetch_unescalated_failures_orders_by_resolved_at(store):
    older = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
    newer = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
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
    a_id = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
    b_id = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
    c_id = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
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
    cursor = await store._db.execute('PRAGMA table_info(tickets)')
    cols = {r[1] for r in await cursor.fetchall()}
    assert 'escalated_at' in cols
    # And idempotent: re-initialising must not raise.
    await store.initialize()


# ---------------------------------------------------------------------------
# extend_pending_expiry — capacity-blocked TTL compensation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extend_pending_expiry_advances_only_pending_in_project(store):
    """Only pending rows in the named project are pushed forward."""
    p1_pending = await store.submit(project_id='p1', candidate_json='{}', ttl_seconds=600)
    p2_pending = await store.submit(project_id='p2', candidate_json='{}', ttl_seconds=600)
    p1_terminal = await store.submit(project_id='p1', candidate_json='{}', ttl_seconds=600)
    await _force_failed(store, p1_terminal, reason='whatever')

    before = {
        p1_pending: (await store.get(p1_pending))['expires_at'],
        p2_pending: (await store.get(p2_pending))['expires_at'],
        p1_terminal: (await store.get(p1_terminal))['expires_at'],
    }

    n = await store.extend_pending_expiry('p1', seconds=100)
    assert n == 1, 'only the single pending p1 row should have been updated'

    after_p1 = (await store.get(p1_pending))['expires_at']
    after_p2 = (await store.get(p2_pending))['expires_at']
    after_p1_terminal = (await store.get(p1_terminal))['expires_at']

    # p1 pending row moved forward by ~100s (sqlite datetime arithmetic
    # rounds to whole seconds for integer additions, but our ".3f" format
    # passes a fractional value through; either way, post > pre).
    pre_dt = datetime.fromisoformat(before[p1_pending])
    post_dt = datetime.fromisoformat(after_p1)
    delta = (post_dt - pre_dt).total_seconds()
    assert 99.0 <= delta <= 101.0, f'expected ~100s shift, got {delta}'

    # The other rows are untouched.
    assert after_p2 == before[p2_pending]
    assert after_p1_terminal == before[p1_terminal]


@pytest.mark.asyncio
@pytest.mark.parametrize('seconds', [0, -1, -100.5])
async def test_extend_pending_expiry_zero_or_negative_is_noop(store, seconds):
    """Non-positive values short-circuit without touching the DB."""
    pid = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
    before = (await store.get(pid))['expires_at']

    n = await store.extend_pending_expiry('p', seconds=seconds)
    assert n == 0

    after = (await store.get(pid))['expires_at']
    assert after == before


# ---------------------------------------------------------------------------
# list_tickets — caller-facing discovery
# ---------------------------------------------------------------------------


async def _force_created_at(store: TicketStore, ticket_id: str, when: datetime) -> None:
    """Test helper: rewrite a ticket's created_at so window filters can be exercised."""
    db = store._db
    await db.execute(
        'UPDATE tickets SET created_at = ? WHERE ticket_id = ?',
        (when.isoformat(), ticket_id),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_list_tickets_default_window_7d(store):
    """Default window (no since=) returns rows newer than now-7d only."""
    now = datetime.now(UTC)
    recent = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
    old = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
    await _force_created_at(store, recent, now - timedelta(days=1))
    await _force_created_at(store, old, now - timedelta(days=8))

    rows = await store.list_tickets('p')
    ids = [r['ticket_id'] for r in rows]
    assert recent in ids
    assert old not in ids


@pytest.mark.asyncio
async def test_list_tickets_status_filter_excludes_other_states(store):
    """status='failed' returns only the failed row."""
    pending = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
    failed = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
    combined = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
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
        tid = await store.submit(project_id='p', candidate_json='{}', ttl_seconds=600)
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
    a_id = await store.submit(project_id='proj-a', candidate_json='{}', ttl_seconds=600)
    b_id = await store.submit(project_id='proj-b', candidate_json='{}', ttl_seconds=600)

    a_rows = await store.list_tickets('proj-a')
    assert {r['ticket_id'] for r in a_rows} == {a_id}
    b_rows = await store.list_tickets('proj-b')
    assert {r['ticket_id'] for r in b_rows} == {b_id}
    await store.close()
