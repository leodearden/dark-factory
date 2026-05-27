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
import logging
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
    db = store._require_db()
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
        )
        b = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(
                title='B', task_id='task-42', escalation_id='esc-42-1',
                suggestion_hash='hash-B',
            ),
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
async def test_worker_dead_marks_pending_tickets_failed_worker_dead(store, tmp_path):
    """When a project's curator worker is dead (no live asyncio.Task and no
    ``_worker_intent`` placeholder), pending tickets for that project must be
    terminalised as ``failed/worker_dead`` and surface as a single
    ``ticket_failure`` escalation grouped by ``(project_id, 'task-curator', _no_escalation_)``.
    """
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        a = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(title='Aa'),
        )
        b = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(title='Bb'),
        )

        janitor = TicketJanitor(
            store, primary_project_root=str(tmp_path),
            liveness_probe=lambda pid: False,  # always dead
        )
        await janitor.tick()

        # Both rows must now be terminal failed/worker_dead.
        for tid in (a, b):
            row = await store.get(tid)
            assert row['status'] == 'failed', f'{tid}: {row}'
            assert row['reason'] == 'worker_dead', f'{tid}: {row["reason"]!r}'

        # And exactly one escalation file must have been emitted (grouped).
        files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(files) == 1, [f.name for f in files]
        body = json.loads(files[0].read_text())
        assert body['category'] == 'ticket_failure'
        assert body['task_id'] == 'task-curator'
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_worker_alive_leaves_pending_alone(store, tmp_path):
    """When the liveness probe reports the project's worker is alive, pending
    tickets must remain pending — the reaper is liveness-gated, not
    timer-gated.
    """
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        a = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(title='still going'),
        )

        janitor = TicketJanitor(
            store, primary_project_root=str(tmp_path),
            liveness_probe=lambda pid: True,  # always alive
        )
        await janitor.tick()

        row = await store.get(a)
        assert row['status'] == 'pending', f'expected pending, got {row}'
        assert row['reason'] is None
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_worker_intent_present_treated_as_alive(store, tmp_path):
    """The ``_worker_intent`` set carried by TaskInterceptor closes the
    spawn-race window between submit_task setting up a queue entry and
    ``_start_worker_if_needed`` actually creating an asyncio.Task: a project
    in the intent set must count as alive even if no Task exists yet."""
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        a = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(title='spawning'),
        )

        intent = {project_id}

        def _liveness(pid: str) -> bool:
            # Simulate: no Task yet, but project has been added to the
            # intent set by submit_task.
            return pid in intent

        janitor = TicketJanitor(
            store, primary_project_root=str(tmp_path),
            liveness_probe=_liveness,
        )
        await janitor.tick()

        row = await store.get(a)
        assert row['status'] == 'pending'
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_worker_dead_escalations_respect_cooldown(store, tmp_path):
    """Two ticks back-to-back with the same dead-worker project must not emit
    duplicate escalations — the existing per-group cooldown still applies to
    rows the reaper terminalises.
    """
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(title='first batch'),
        )

        janitor = TicketJanitor(
            store,
            cooldown_secs=3600.0,
            primary_project_root=str(tmp_path),
            liveness_probe=lambda pid: False,
        )
        await janitor.tick()
        first = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(first) == 1

        # Second batch lands while the cooldown window is still open.
        b = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(title='second batch'),
        )
        await janitor.tick()

        # Still one escalation, but b must be stamped so a third tick doesn't
        # re-evaluate it.
        second = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(second) == 1, 'cooldown must suppress the re-escalation'
        row_b = await store.get(b)
        assert row_b['status'] == 'failed'
        assert row_b['reason'] == 'worker_dead'
        assert row_b['escalated_at'] is not None
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_janitor_accepts_known_projects_kwarg_and_uses_injected_map(store, tmp_path):
    """Injected known_projects map drives tick() resolution even when primary_project_root=''.

    Uses primary_project_root='' deliberately so the only way to resolve
    project_root is via the injected map. An escalation file landing at
    tmp_path/data/escalations/ proves the map was used.
    """
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        ticket_id = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(task_id='t1', escalation_id='esc-di-1'),
        )
        await _force_failed(store, ticket_id, reason='curator_rejected')

        janitor = TicketJanitor(
            store,
            primary_project_root='',
            known_projects={project_id: str(tmp_path)},
        )
        await janitor.tick()

        files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(files) == 1, (
            f'Expected exactly one escalation file via injected map; got {[f.name for f in files]}'
        )
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_janitor_default_known_projects_kwarg_falls_back_to_build_known_projects_map(
    store, tmp_path
):
    """When known_projects kwarg is omitted, janitor falls back to build_known_projects_map.

    Verifies back-compat: existing tests that pass primary_project_root still work.
    """
    from fused_memory.models.scope import build_known_projects_map

    janitor = TicketJanitor(store, primary_project_root=str(tmp_path))
    expected = build_known_projects_map(str(tmp_path))
    assert janitor._known_projects == expected


@pytest.mark.asyncio
async def test_init_snapshots_known_projects_against_post_init_env_mutation(
    store, tmp_path, monkeypatch
):
    """_known_projects is frozen at __init__ time; post-init env mutations have no effect.

    Guards the snapshot contract from task 1164: the registry is built once at
    construction and never rebuilt on tick(), so DASHBOARD_KNOWN_PROJECT_ROOTS
    changes require a restart to take effect.
    """
    proj_a = tmp_path / 'proj_a'
    proj_b = tmp_path / 'proj_b'
    proj_a.mkdir()
    proj_b.mkdir()

    monkeypatch.setenv('DASHBOARD_KNOWN_PROJECT_ROOTS', str(proj_a))
    janitor = TicketJanitor(store, primary_project_root='')

    pre_mutation = dict(janitor._known_projects)
    proj_a_id = _project_id_for(proj_a)
    assert pre_mutation, 'registry must be non-empty after init with env var set'
    assert proj_a_id in pre_mutation, (
        f'proj_a project_id {proj_a_id!r} must appear in registry; got {pre_mutation}'
    )

    # Submit a failed ticket so tick() has a row to process, forcing it through
    # the _known_projects.get() code path.  Without rows tick() returns early
    # and never touches the registry — the snapshot contract would be untested.
    ticket_id = await store.submit(
        project_id=proj_a_id,
        candidate_json=_candidate_blob(task_id='t1', escalation_id='esc-1'),
    )
    await _force_failed(store, ticket_id, reason='curator_rejected')

    monkeypatch.setenv('DASHBOARD_KNOWN_PROJECT_ROOTS', str(proj_b))

    # tick() must route the failed ticket using the snapshotted registry, not
    # the current env.  The proj_a orchestrator lock doesn't exist so the
    # escalation is skipped, but the _known_projects lookup itself is exercised.
    await janitor.tick()

    proj_b_id = _project_id_for(proj_b)
    assert proj_b_id not in janitor._known_projects, (
        'post-init env mutation must not leak into snapshotted registry'
    )
    assert janitor._known_projects == pre_mutation, (
        'post-init env mutation must not change the janitor registry; '
        f'registry changed from {pre_mutation} to {janitor._known_projects}'
    )


@pytest.mark.asyncio
async def test_repeated_probe_raises_surface_infra_issue_escalation(store, tmp_path):
    """Consecutive probe raises accumulate and surface an infra_issue at the threshold.

    Below the threshold (3) no escalation is emitted.  At the threshold,
    exactly one infra_issue escalation is submitted.  The pending ticket must
    remain status='pending' throughout — fail-open is preserved; the reaper
    never fires and the ticket is never stamped.
    """
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        ticket_id = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(title='stranded task'),
        )

        def _always_raises(pid: str) -> bool:
            raise RuntimeError('probe broken')

        janitor = TicketJanitor(
            store,
            primary_project_root=str(tmp_path),
            liveness_probe=_always_raises,
            probe_defect_threshold=3,
        )

        esc_dir = tmp_path / 'data' / 'escalations'

        # Tick 1 — below threshold, no escalation
        await janitor.tick()
        files = sorted(esc_dir.glob('esc-*.json')) if esc_dir.exists() else []
        assert files == [], (
            f'Expected no escalation after 1st tick; got {[f.name for f in files]}'
        )
        row = await store.get(ticket_id)
        assert row['status'] == 'pending', f'ticket must stay pending after 1st tick; got {row}'
        assert row['reason'] is None

        # Tick 2 — below threshold, still no escalation
        await janitor.tick()
        files = sorted(esc_dir.glob('esc-*.json')) if esc_dir.exists() else []
        assert files == [], (
            f'Expected no escalation after 2nd tick; got {[f.name for f in files]}'
        )
        row = await store.get(ticket_id)
        assert row['status'] == 'pending', f'ticket must stay pending after 2nd tick; got {row}'
        assert row['reason'] is None

        # Tick 3 — at threshold, exactly one infra_issue escalation surfaced
        await janitor.tick()
        files = sorted(esc_dir.glob('esc-*.json'))
        assert len(files) == 1, (
            f'Expected exactly 1 escalation after 3rd tick; got {[f.name for f in files]}'
        )
        body = json.loads(files[0].read_text())
        assert body['category'] == 'infra_issue'
        assert body['severity'] == 'info'
        assert body['agent_role'] == 'fused-memory/ticket-janitor'
        # Fail-open preserved: ticket is still pending, never reaped, never stamped
        row = await store.get(ticket_id)
        assert row['status'] == 'pending', (
            f'Ticket must stay pending (fail-open); got {row}'
        )
        assert row['reason'] is None
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_probe_defect_escalation_is_rate_limited(store, tmp_path):
    """Post-threshold probe raises must not flood the escalation queue.

    With probe_defect_threshold=1 the first raise surfaces immediately.
    Subsequent raises within the cooldown window must be suppressed so that
    only a single esc-*.json file exists across multiple ticks.  The pending
    ticket remains status='pending' throughout.
    """
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        ticket_id = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(title='still stranded'),
        )

        def _always_raises(pid: str) -> bool:
            raise RuntimeError('probe still broken')

        janitor = TicketJanitor(
            store,
            primary_project_root=str(tmp_path),
            liveness_probe=_always_raises,
            probe_defect_threshold=1,
            cooldown_secs=3600.0,
        )

        esc_dir = tmp_path / 'data' / 'escalations'

        # 4 ticks — first one surfaces at threshold=1, rest must be suppressed
        for _ in range(4):
            await janitor.tick()

        files = sorted(esc_dir.glob('esc-*.json'))
        assert len(files) == 1, (
            f'Cooldown must suppress duplicate probe-defect escalations; '
            f'got {[f.name for f in files]}'
        )
        # Ticket is still pending the entire time
        row = await store.get(ticket_id)
        assert row['status'] == 'pending'
        assert row['reason'] is None
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_startup_nudge_emitted_once_across_two_constructions(
    store, tmp_path, caplog, monkeypatch
):
    """The startup INFO nudge fires exactly once per process regardless of
    how many TicketJanitor instances are constructed.

    Guards the once-per-process semantics from task 1210: the class-level
    ``_registry_log_emitted`` flag must be set to True after the first
    construction and suppress the log in all subsequent ones.  Resetting
    the flag at the start of this test makes the assertion order-independent
    — it does not matter which earlier test in the module already tripped
    the flag.
    """
    # Reset via monkeypatch so pytest restores the original value automatically,
    # even if an assertion fails mid-test.
    monkeypatch.setattr(TicketJanitor, '_registry_log_emitted', False)
    caplog.set_level(logging.INFO, logger='fused_memory.middleware.ticket_janitor')

    j1 = TicketJanitor(store, primary_project_root=str(tmp_path))
    # Guard's side-effect: flag must be True after the first construction.
    assert TicketJanitor._registry_log_emitted is True

    TicketJanitor(store, primary_project_root=str(tmp_path))  # second construction must not re-emit

    nudge_records = [
        r for r in caplog.records
        if 'project registry snapshotted at init' in r.getMessage()
    ]
    assert len(nudge_records) == 1, (
        f'Expected exactly 1 startup nudge across 2 constructions; '
        f'got {len(nudge_records)}: {[r.getMessage() for r in nudge_records]}'
    )
    # Verify the %d argument was actually passed to the logger (not pre-formatted
    # into the message string) and equals the project count.  LogRecord.args[0]
    # gives the raw integer directly — no string parsing, no digit collisions
    # with unrelated tokens in the log message (e.g. task references).
    assert nudge_records[0].args[0] == len(j1._known_projects), (
        f'Expected logger arg {len(j1._known_projects)} but got '
        f'{nudge_records[0].args[0]!r}'
    )


