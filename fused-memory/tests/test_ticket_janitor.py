"""Tests for :mod:`fused_memory.middleware.ticket_janitor`.

The janitor sweeps failed tickets out of :class:`TicketStore`, batches them
by ``(project_id, task_id, escalation_id)`` and submits one info-severity
``ticket_failure`` escalation per batch. These tests exercise:

* Grouping by metadata + correct row-stamping after submit.
* Cooldown stamping that suppresses re-escalation of the same group.
* Pass-through of ``failed/server_restart`` rows.
* Fallback for ``failed/bad_candidate_json`` (unparseable JSON).
* No-orchestrator path: tickets stay un-escalated and retry next tick.
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import IO, Literal, overload

import pytest
import pytest_asyncio

from fused_memory.middleware.ticket_janitor import TicketJanitor
from fused_memory.middleware.ticket_store import TicketStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = TicketStore(tmp_path / 'tickets.db')
    await s.initialize()
    yield s
    await s.close()


@overload
def _make_orchestrator_layout(root, *, hold_lock: Literal[True]) -> IO[bytes]: ...
@overload
def _make_orchestrator_layout(root, *, hold_lock: Literal[False]) -> None: ...
def _make_orchestrator_layout(root, *, hold_lock: bool) -> IO[bytes] | None:
    """Create ``data/orchestrator/orchestrator.lock`` and optionally hold LOCK_EX.

    Mirrors the helper in test_curator_escalator.py so the janitor's
    liveness probe sees the same shape as the curator escalator's.
    """
    lock_dir = root / 'data' / 'orchestrator'
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / 'orchestrator.lock'
    lock_path.write_text('')
    if not hold_lock:
        return None
    handle = lock_path.open('r+b')
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def _project_id_for(root: Path) -> str:
    """Reproduce :func:`scope.resolve_project_id` for test setup."""
    return root.name.lower().replace('-', '_')


def _candidate_blob(
    *,
    title: str = 'A task',
    task_id: str | None = None,
    escalation_id: str | None = None,
    suggestion_hash: str | None = None,
) -> str:
    """Build a synthetic candidate_json that mirrors the interceptor's format."""
    metadata: dict = {}
    if task_id is not None:
        metadata['task_id'] = task_id
    if escalation_id is not None:
        metadata['escalation_id'] = escalation_id
    if suggestion_hash is not None:
        metadata['suggestion_hash'] = suggestion_hash
    return json.dumps({
        'project_root': '/dummy',
        'kwargs': {'title': title},
        'metadata': metadata,
    })


async def _force_failed(store: TicketStore, ticket_id: str, *, reason: str) -> None:
    db = store._db
    await db.execute(
        "UPDATE tickets SET status='failed', reason=?, resolved_at=datetime('now') "
        "WHERE ticket_id=?",
        (reason, ticket_id),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_groups_failures_and_emits_one_escalation_per_group(store, tmp_path):
    """Two failed tickets sharing (task, escalation) → 1 escalation; rows stamped."""
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        # Two rows in the same group.
        a = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(
                title='A', task_id='task-42', escalation_id='esc-42-1',
                suggestion_hash='hash-A',
            ),
            ttl_seconds=600,
        )
        b = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(
                title='B', task_id='task-42', escalation_id='esc-42-1',
                suggestion_hash='hash-B',
            ),
            ttl_seconds=600,
        )
        await _force_failed(store, a, reason='curator_rejected')
        await _force_failed(store, b, reason='curator_rejected')

        janitor = TicketJanitor(store, primary_project_root=str(tmp_path))
        await janitor.tick()

        files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(files) == 1, [f.name for f in files]
        body = json.loads(files[0].read_text())
        assert body['category'] == 'ticket_failure'
        assert body['severity'] == 'info'
        assert body['agent_role'] == 'fused-memory/ticket-janitor'
        assert body['task_id'] == 'task-42'
        # Detail JSON-encodes the per-row payload, which the steward reads.
        detail = json.loads(body['detail'])
        assert {row['ticket_id'] for row in detail} == {a, b}
        for row in detail:
            assert row['suggestion_hash'] in {'hash-A', 'hash-B'}

        # Both rows must be stamped so a follow-up tick doesn't re-escalate.
        for tid in (a, b):
            row = await store.get(tid)
            assert row['escalated_at'] is not None
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_separate_escalations_for_distinct_groups(store, tmp_path):
    """Two tickets with different escalation_ids → two queue files."""
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        a = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(task_id='t1', escalation_id='esc-1'),
        )
        b = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(task_id='t1', escalation_id='esc-2'),
        )
        await _force_failed(store, a, reason='r')
        await _force_failed(store, b, reason='r')

        janitor = TicketJanitor(store, primary_project_root=str(tmp_path))
        await janitor.tick()

        files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(files) == 2
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_cooldown_stamps_rows_without_emitting_escalation(store, tmp_path):
    """Within the cooldown, a fresh group still gets ``escalated_at`` stamped
    so the next tick doesn't re-evaluate it — accepted loss-of-signal."""
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        # First batch: emits one escalation.
        a = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(task_id='t', escalation_id='e'),
        )
        await _force_failed(store, a, reason='r')

        janitor = TicketJanitor(
            store,
            cooldown_secs=3600.0,
            primary_project_root=str(tmp_path),
        )
        await janitor.tick()
        first = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(first) == 1

        # Second batch (same group) within the cooldown.
        b = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(task_id='t', escalation_id='e'),
        )
        await _force_failed(store, b, reason='r')

        await janitor.tick()
        second = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(second) == 1, 'cooldown must suppress the re-escalation'
        # …but `b` must be stamped so a third tick doesn't re-evaluate it.
        row = await store.get(b)
        assert row['escalated_at'] is not None
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_no_orchestrator_leaves_rows_for_retry(store, tmp_path):
    """Liveness probe negative → log + skip; rows stay un-stamped."""
    # Lock file exists but no exclusive holder.
    _make_orchestrator_layout(tmp_path, hold_lock=False)
    project_id = _project_id_for(tmp_path)

    a = await store.submit(
        project_id=project_id,
        candidate_json=_candidate_blob(task_id='t', escalation_id='e'),
    )
    await _force_failed(store, a, reason='r')

    janitor = TicketJanitor(store, primary_project_root=str(tmp_path))
    await janitor.tick()

    files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
    assert files == [], 'no orchestrator → no escalation submitted'
    row = await store.get(a)
    assert row['escalated_at'] is None, 'row must remain unstamped for retry'


@pytest.mark.asyncio
async def test_server_restart_rows_get_escalated(store, tmp_path):
    """Rows produced by ``flush_pending_on_startup`` are picked up normally."""
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        a = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(task_id='t', escalation_id='e'),
        )
        # Simulate a clean restart: flush_pending_on_startup() turns this row
        # into status='failed' / reason='server_restart'.
        n = await store.flush_pending_on_startup()
        assert n == 1

        janitor = TicketJanitor(store, primary_project_root=str(tmp_path))
        await janitor.tick()

        files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(files) == 1
        body = json.loads(files[0].read_text())
        detail = json.loads(body['detail'])
        assert detail[0]['ticket_id'] == a
        assert detail[0]['reason'] == 'server_restart'
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_bad_candidate_json_falls_back_to_unparseable_group(store, tmp_path):
    """Rows whose candidate_json is unparseable still emit one escalation."""
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        a = await store.submit(
            project_id=project_id,
            candidate_json='not valid json',
        )
        await _force_failed(store, a, reason='bad_candidate_json')

        janitor = TicketJanitor(store, primary_project_root=str(tmp_path))
        await janitor.tick()

        files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(files) == 1
        body = json.loads(files[0].read_text())
        # Falls back to the curator-bucket task_id rather than a real task id.
        assert body['task_id'] == 'task-curator'
        assert 'unparseable' in body['summary']
        # Row stamped so it doesn't re-escalate.
        row = await store.get(a)
        assert row['escalated_at'] is not None
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_idempotency_hit_rows_excluded(store, tmp_path):
    """combined/idempotency_hit is happy-path — never escalated."""
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        a = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(task_id='t', escalation_id='e'),
        )
        # Force a failed-status row with the idempotency-hit reason. (Real
        # idempotency hits land as status='combined' which is also excluded
        # by the status='failed' filter; this synthetic case verifies the
        # belts-and-braces ``reason`` exclusion.)
        await _force_failed(store, a, reason='idempotency_hit')

        janitor = TicketJanitor(store, primary_project_root=str(tmp_path))
        await janitor.tick()

        files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert files == []
        # And the row must remain un-stamped so the explicit filter keeps
        # excluding it (semantic: it's healthy, not "handled").
        row = await store.get(a)
        assert row['escalated_at'] is None
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_sweep_expired_runs_first(store, tmp_path):
    """Pending-but-expired rows graduate to failed/expired in the same tick."""
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        a = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(task_id='t', escalation_id='e'),
            ttl_seconds=600,
        )
        # Backdate expires_at so sweep_expired marks it failed.
        db = store._db
        await db.execute(
            "UPDATE tickets SET expires_at='2020-01-01T00:00:00+00:00' "
            "WHERE ticket_id=?",
            (a,),
        )
        await db.commit()

        janitor = TicketJanitor(store, primary_project_root=str(tmp_path))
        await janitor.tick()

        # Expired row terminalised AND escalated in one pass.
        row = await store.get(a)
        assert row['status'] == 'failed'
        assert row['reason'] == 'expired'
        files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(files) == 1
    finally:
        handle.close()
