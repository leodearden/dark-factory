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
from _fm_helpers import LoopFreedomProbe

from fused_memory.middleware import ticket_janitor
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
    db = store._require_access().connection
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
        # Count and project_id must round-trip through detail and summary.
        detail = json.loads(body['detail'])
        assert detail['consecutive_probe_failures'] == 3, (
            f'Expected consecutive_probe_failures=3 in detail; got {detail}'
        )
        assert detail['project_id'] == project_id, (
            f'Expected project_id={project_id!r} in detail; got {detail}'
        )
        assert str(3) in body['summary'], (
            f'Expected "3" in summary to confirm count was serialised; got {body["summary"]!r}'
        )
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
async def test_probe_success_resets_consecutive_failure_counter(store, tmp_path):
    """A probe success resets the per-project counter so intermittent blips never accumulate.

    A probe that alternates raise/succeed/raise/succeed... never builds up
    3 consecutive raises and must therefore never surface an escalation, even
    across many ticks.  The pending ticket stays pending throughout.
    """
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        ticket_id = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(title='intermittent probe'),
        )

        call_count = [0]

        def _alternating_probe(pid: str) -> bool:
            call_count[0] += 1
            if call_count[0] % 2 == 1:  # odd calls raise
                raise RuntimeError('intermittent probe failure')
            return True  # even calls succeed

        janitor = TicketJanitor(
            store,
            primary_project_root=str(tmp_path),
            liveness_probe=_alternating_probe,
            probe_defect_threshold=3,
        )

        esc_dir = tmp_path / 'data' / 'escalations'

        # 6 ticks: alternating raise/succeed — consecutive count never reaches 3
        for _ in range(6):
            await janitor.tick()

        files = sorted(esc_dir.glob('esc-*.json')) if esc_dir.exists() else []
        assert files == [], (
            f'Intermittent probe raises must never surface an escalation; '
            f'got {[f.name for f in files]}'
        )
        row = await store.get(ticket_id)
        assert row['status'] == 'pending'
        assert row['reason'] is None
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_probe_defect_bail_out_when_orchestrator_not_running(store, tmp_path):
    """_surface_probe_defect bail-out when the orchestrator is not running.

    Guards the branches of _surface_probe_defect that return early before
    queue.submit — analogous to the no-orchestrator path in tick() step 4.

    Invariants when the orchestrator is absent:
    (a) No esc-*.json file is written (bail-out before submit).
    (b) The per-project failure counter is NOT reset — it keeps accumulating
        so the defect will be surfaced once the orchestrator starts.
    (c) _escalation_log[(pid, _PROBE_DEFECT, _PROBE_DEFECT)] stays empty so
        there is no cooldown recorded and the next tick retries immediately.
    """
    from fused_memory.middleware.ticket_janitor import _PROBE_DEFECT

    # Create the lock file but do NOT hold it — is_orchestrator_lock_held returns False.
    _make_orchestrator_layout(tmp_path, hold_lock=False)
    project_id = _project_id_for(tmp_path)
    ticket_id = await store.submit(
        project_id=project_id,
        candidate_json=_candidate_blob(title='bail-out probe test'),
    )

    def _always_raises(pid: str) -> bool:
        raise RuntimeError('probe broken')

    # threshold=1: the very first raise should try to surface — but bail-out
    # because the orchestrator is not running.
    janitor = TicketJanitor(
        store,
        primary_project_root=str(tmp_path),
        liveness_probe=_always_raises,
        probe_defect_threshold=1,
    )

    for _ in range(3):
        await janitor.tick()

    esc_dir = tmp_path / 'data' / 'escalations'
    files = sorted(esc_dir.glob('esc-*.json')) if esc_dir.exists() else []

    # (a) No escalation file submitted.
    assert files == [], (
        f'Expected no escalation when orchestrator is not running; '
        f'got {[f.name for f in files]}'
    )
    # (b) Counter keeps accumulating — not reset by the bail-out.
    assert janitor._probe_failures[project_id] == 3, (
        f'Counter must accumulate (not reset) on bail-out; '
        f'got {janitor._probe_failures[project_id]}'
    )
    # (c) Cooldown log stays empty — no cooldown recorded so next tick retries.
    key = (project_id, _PROBE_DEFECT, _PROBE_DEFECT)
    assert janitor._escalation_log[key] == [], (
        f'Cooldown log must stay empty on bail-out; '
        f'got {janitor._escalation_log[key]}'
    )
    # Ticket is still pending throughout (fail-open preserved).
    row = await store.get(ticket_id)
    assert row['status'] == 'pending'
    assert row['reason'] is None


def _escalation_categories(root: Path) -> list[str]:
    esc_dir = root / 'data' / 'escalations'
    return [
        json.loads(f.read_text())['category']
        for f in sorted(esc_dir.glob('esc-*.json'))
    ]


@pytest.mark.asyncio
async def test_ticket_failure_path_probes_orchestrator_off_the_event_loop(
    store, tmp_path, monkeypatch,
):
    """tick()'s ticket_failure path runs the orchestrator probe off the loop thread."""
    probe = LoopFreedomProbe()
    seen: list[str] = []

    def _blocking_lock_probe(project_root):
        seen.append(project_root)
        probe.block()
        return True

    monkeypatch.setattr(ticket_janitor, 'is_orchestrator_lock_held', _blocking_lock_probe)
    ticket_id = await store.submit(
        project_id=_project_id_for(tmp_path),
        candidate_json=_candidate_blob(task_id='task-7', escalation_id='esc-7-1'),
    )
    await _force_failed(store, ticket_id, reason='curator_rejected')

    await TicketJanitor(store, primary_project_root=str(tmp_path)).tick()

    probe.assert_loop_stayed_free()
    assert seen == [str(tmp_path)]
    assert _escalation_categories(tmp_path) == ['ticket_failure']


@pytest.mark.asyncio
async def test_probe_defect_path_probes_orchestrator_off_the_event_loop(
    store, tmp_path, monkeypatch,
):
    """The probe-defect path runs the orchestrator probe off the loop thread."""
    probe = LoopFreedomProbe()

    def _blocking_lock_probe(project_root):
        probe.block()
        return True

    def _raising_liveness_probe(pid: str) -> bool:
        raise RuntimeError('probe broken')

    monkeypatch.setattr(ticket_janitor, 'is_orchestrator_lock_held', _blocking_lock_probe)
    await store.submit(
        project_id=_project_id_for(tmp_path),
        candidate_json=_candidate_blob(title='stranded task'),
    )
    janitor = TicketJanitor(
        store,
        primary_project_root=str(tmp_path),
        liveness_probe=_raising_liveness_probe,
        probe_defect_threshold=1,
    )

    await janitor.tick()

    probe.assert_loop_stayed_free()
    assert _escalation_categories(tmp_path) == ['infra_issue']


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


@pytest.mark.asyncio
async def test_worker_dead_reap_invokes_signal_callback_per_ticket(store, tmp_path):
    """When the reaper terminalises pending tickets for a dead worker, the
    injected signal_ticket_resolved callback is invoked exactly once per
    reaped ticket id — no more, no less.

    RED until TicketJanitor accepts signal_ticket_resolved and invokes it.
    """
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        a = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(title='Ticket-A'),
        )
        b = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(title='Ticket-B'),
        )

        signalled: list[str] = []
        janitor = TicketJanitor(
            store,
            primary_project_root=str(tmp_path),
            liveness_probe=lambda pid: False,  # always dead
            signal_ticket_resolved=signalled.append,
        )
        await janitor.tick()

        # Callback must have been called once per reaped ticket id.
        assert set(signalled) == {a, b}, (
            f'Expected callback for both ticket ids; got {signalled}'
        )
        assert len(signalled) == 2, (
            f'Expected exactly 2 callback invocations; got {len(signalled)}'
        )

        # Rows are terminal failed/worker_dead.
        for tid in (a, b):
            row = await store.get(tid)
            assert row['status'] == 'failed', f'{tid}: {row}'
            assert row['reason'] == 'worker_dead', f'{tid}: {row["reason"]!r}'
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_worker_dead_reap_no_signal_callback_is_safe(store, tmp_path):
    """With the default signal_ticket_resolved=None, the reaper still
    terminalises pending tickets and tick() does not raise — back-compat.
    """
    handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
    try:
        project_id = _project_id_for(tmp_path)
        a = await store.submit(
            project_id=project_id,
            candidate_json=_candidate_blob(title='Ticket-A'),
        )

        # signal_ticket_resolved defaults to None — must not raise.
        janitor = TicketJanitor(
            store,
            primary_project_root=str(tmp_path),
            liveness_probe=lambda pid: False,
        )
        await janitor.tick()  # must not raise

        row = await store.get(a)
        assert row['status'] == 'failed'
        assert row['reason'] == 'worker_dead'
    finally:
        handle.close()


class TestDedupOutageDetector:
    """The janitor escalates the dedup-outage signature: a window of curated
    tickets with no combine at all, resolved faster than a real curator LLM
    call can complete (task 4718).

    The detector walks the injected project registry, so an unresolvable
    project_root cannot arise for it; its routing bail-outs are the
    escalation package being absent, the orchestrator not running, and the
    submit raising.
    """

    _OUTAGE_REASON = 'create: llm-failed: FileNotFoundError: x'

    @pytest.fixture
    def pid(self, tmp_path):
        return _project_id_for(tmp_path)

    @pytest.fixture
    def lock(self, tmp_path):
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        yield handle
        handle.close()

    @staticmethod
    def _janitor(store, tmp_path, pid, cfg=None, **kwargs):
        from fused_memory.config.schema import DedupOutageDetectorConfig

        return TicketJanitor(
            store,
            known_projects={pid: str(tmp_path)},
            dedup_outage=cfg if cfg is not None else DedupOutageDetectorConfig(),
            **kwargs,
        )

    @staticmethod
    async def _seed(store, tmp_path, pid, *, created, combined=0, latency_s=2.0,
                    reason=_OUTAGE_REASON):
        from datetime import UTC, datetime, timedelta

        from _fm_helpers import seed_resolved_ticket

        resolved_at = datetime.now(UTC) - timedelta(minutes=5)
        for _ in range(created):
            await seed_resolved_ticket(
                store, tmp_path / 'tickets.db', pid, status='created',
                latency_s=latency_s, resolved_at=resolved_at, reason=reason,
            )
        for _ in range(combined):
            await seed_resolved_ticket(
                store, tmp_path / 'tickets.db', pid, status='combined',
                latency_s=latency_s, resolved_at=resolved_at, reason='combine: x',
            )

    @staticmethod
    def _escalations(root: Path) -> list[dict]:
        esc_dir = root / 'data' / 'escalations'
        if not esc_dir.exists():
            return []
        return [json.loads(f.read_text()) for f in sorted(esc_dir.glob('esc-*.json'))]

    @pytest.mark.asyncio
    async def test_fires_one_blocking_infra_escalation_naming_the_reason(
        self, store, tmp_path, pid, lock,
    ):
        await self._seed(store, tmp_path, pid, created=10)

        await self._janitor(store, tmp_path, pid).tick()

        escalations = self._escalations(tmp_path)
        assert len(escalations) == 1, escalations
        body = escalations[0]
        assert body['category'] == 'infra_issue'
        assert body['agent_role'] == 'fused-memory/ticket-janitor'
        assert body['task_id'] == 'task-curator'
        assert body['level'] == 1
        assert body['severity'] == 'blocking'
        assert pid in body['summary']
        assert '10 tickets resolved' in body['summary']
        assert '0 combined' in body['summary']
        assert 'median resolve 2.0s' in body['summary']
        detail = json.loads(body['detail'])
        assert detail['project_id'] == pid
        assert detail['window_seconds'] == 21600.0
        assert detail['resolved'] == 10
        assert detail['combined'] == 0
        assert detail['median_resolve_seconds'] == pytest.approx(2.0, abs=1e-3)
        assert detail['top_create_reasons'] == [[self._OUTAGE_REASON, 10]]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('created', 'combined', 'latency_s'),
        [(10, 1, 2.0), (10, 0, 100.0), (9, 0, 2.0)],
        ids=['one-combine', 'slow-median', 'too-few-samples'],
    )
    async def test_does_not_fire_without_the_full_signature(
        self, store, tmp_path, pid, lock, created, combined, latency_s,
    ):
        await self._seed(
            store, tmp_path, pid, created=created, combined=combined, latency_s=latency_s,
        )

        await self._janitor(store, tmp_path, pid).tick()

        assert self._escalations(tmp_path) == []

    @pytest.mark.asyncio
    async def test_disabled_or_omitted_config_does_not_fire(self, store, tmp_path, pid, lock):
        from fused_memory.config.schema import DedupOutageDetectorConfig

        await self._seed(store, tmp_path, pid, created=10)

        await self._janitor(
            store, tmp_path, pid, cfg=DedupOutageDetectorConfig(enabled=False),
        ).tick()
        await TicketJanitor(store, known_projects={pid: str(tmp_path)}).tick()

        assert self._escalations(tmp_path) == []

    @pytest.mark.asyncio
    async def test_rate_limited_on_its_own_window_length(
        self, store, tmp_path, pid, lock, monkeypatch,
    ):
        import types

        clock = [1_000_000.0]
        monkeypatch.setattr(
            ticket_janitor, 'time', types.SimpleNamespace(monotonic=lambda: clock[0]),
        )
        await self._seed(store, tmp_path, pid, created=10)
        janitor = self._janitor(store, tmp_path, pid, cooldown_secs=3600.0)

        await janitor.tick()
        await janitor.tick()
        clock[0] += 3601.0
        await janitor.tick()
        assert len(self._escalations(tmp_path)) == 1, (
            'a sustained outage must not re-escalate inside window_seconds, '
            'even past the janitor-wide cooldown'
        )

        clock[0] += 21600.0
        await janitor.tick()
        assert len(self._escalations(tmp_path)) == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize('bail_out', ['no-escalation-package', 'submit-raises'])
    async def test_bail_out_records_no_rate_limit(
        self, store, tmp_path, pid, lock, monkeypatch, bail_out,
    ):
        await self._seed(store, tmp_path, pid, created=10)
        janitor = self._janitor(store, tmp_path, pid)

        def _raise(*_args, **_kwargs):
            raise OSError('disk full')

        with monkeypatch.context() as m:
            if bail_out == 'no-escalation-package':
                m.setattr(ticket_janitor, 'HAS_ESCALATION', False)
            else:
                m.setattr(ticket_janitor.EscalationQueue, 'submit', _raise)
            await janitor.tick()
        assert self._escalations(tmp_path) == []

        await janitor.tick()
        assert len(self._escalations(tmp_path)) == 1

    @pytest.mark.asyncio
    async def test_orchestrator_not_running_records_no_rate_limit(self, store, tmp_path, pid):
        _make_orchestrator_layout(tmp_path, hold_lock=False)
        await self._seed(store, tmp_path, pid, created=10)
        janitor = self._janitor(store, tmp_path, pid)

        await janitor.tick()
        assert self._escalations(tmp_path) == []

        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            await janitor.tick()
        finally:
            handle.close()
        assert len(self._escalations(tmp_path)) == 1

    @pytest.mark.asyncio
    async def test_raising_health_read_leaves_the_failure_sweep_running(
        self, store, tmp_path, pid, lock, monkeypatch,
    ):
        from unittest.mock import AsyncMock

        ticket_id = await store.submit(
            project_id=pid,
            candidate_json=_candidate_blob(task_id='task-7', escalation_id='esc-7-1'),
        )
        await _force_failed(store, ticket_id, reason='curator_rejected')
        monkeypatch.setattr(
            store, 'dedup_health', AsyncMock(side_effect=RuntimeError('db gone')),
        )

        await self._janitor(store, tmp_path, pid).tick()

        assert _escalation_categories(tmp_path) == ['ticket_failure']

    @pytest.mark.asyncio
    async def test_detector_and_failure_sweep_both_run_in_one_tick(
        self, store, tmp_path, pid, lock,
    ):
        await self._seed(store, tmp_path, pid, created=10)
        ticket_id = await store.submit(
            project_id=pid,
            candidate_json=_candidate_blob(task_id='task-7', escalation_id='esc-7-1'),
        )
        await _force_failed(store, ticket_id, reason='curator_rejected')

        await self._janitor(store, tmp_path, pid).tick()

        assert sorted(_escalation_categories(tmp_path)) == ['infra_issue', 'ticket_failure']

    @pytest.mark.asyncio
    async def test_each_project_is_judged_independently(self, store, tmp_path):
        from fused_memory.config.schema import DedupOutageDetectorConfig

        roots = {name: tmp_path / name for name in ('alpha', 'beta')}
        handles = [_make_orchestrator_layout(root, hold_lock=True) for root in roots.values()]
        try:
            await self._seed(store, tmp_path, 'alpha', created=10)
            await self._seed(store, tmp_path, 'beta', created=10, combined=3, latency_s=90.0)
            janitor = TicketJanitor(
                store,
                known_projects={pid: str(root) for pid, root in roots.items()},
                dedup_outage=DedupOutageDetectorConfig(),
            )

            await janitor.tick()
        finally:
            for handle in handles:
                handle.close()

        assert len(self._escalations(roots['alpha'])) == 1
        assert self._escalations(roots['beta']) == []
