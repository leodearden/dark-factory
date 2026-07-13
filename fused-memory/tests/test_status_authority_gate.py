"""ζ integration gate — A-rows (orchestrator <-> fused-memory store seam).

PRD ``plans/task-status-authority-prd.md`` §Boundary-test sketch, rows
A1-A6 of the 17-cell boundary matrix. This module realizes those rows as
TWO-WAY tests that assert through the product's own read paths
(``get_task``/``get_statuses``) rather than mocking the taskmaster and
checking ``mock.assert_called_once()`` — the per-task unit tests
(``test_sqlite_task_backend.py``, ``test_task_interceptor.py``) already
cover each cell in isolation with a mocked taskmaster; this gate wires a
REAL :class:`SqliteTaskBackend` behind a REAL :class:`TaskInterceptor` so a
regression in the vocabulary gate (rho1a), the transition-legality gate
(rho1b), or the claimant/is_stranded plumbing (rho2/omega1) trips a single
consolidated CI signal.

A1-A4 (vocabulary rejection + transition-legality log/enforce modes) drive
the interceptor via :func:`_fresh_stack`, which wires a fresh backend behind
a fresh interceptor per call — *enforce* toggles log-mode
(``config=None``) vs enforce-mode (``FusedMemoryConfig`` with
``task_status.enforce_transitions=True``), both over the SAME real backend
so a test can assert the persisted row either way.

A5-A6 (claimant round-trip + is_stranded over a backend-fetched row) drive
the backend directly via the ``backend``/``project_root`` fixtures — the
claimant/heartbeat plumbing they exercise is native to
:class:`SqliteTaskBackend` (``set_task_status``/``set_task_claimant``/
``get_task``) and has no transition-legality gate to toggle, so no
interceptor is needed (mirrors ``test_sqlite_task_backend.py``:407-553).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from shared.task_claimant import compose_claimant_run_id, is_stranded
from shared.task_statuses import TaskStatus

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.backends.task_backend_errors import TaskmasterError
from fused_memory.config.schema import FusedMemoryConfig, TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer


async def _fresh_stack(tmp_path, *, enforce: bool = False):
    """Open a brand-new ``(interceptor, backend, project_root, event_buffer)``
    stack rooted at *tmp_path*, mirroring
    ``test_candidate_key_boundary.py``'s ``open_fresh_stack``.

    *enforce* toggles the interceptor's transition-legality gate (Table A,
    task 2175/rho1b): ``False`` (default) wires ``config=None`` — log-mode,
    an illegal transition WARNs and the write proceeds; ``True`` wires a
    ``FusedMemoryConfig`` with ``task_status.enforce_transitions=True`` —
    enforce-mode, an illegal transition is typed-rejected with no write.
    Both modes run over a REAL :class:`SqliteTaskBackend` so A1-A4 can
    assert the persisted row (``get_task``/``get_statuses``) either way.

    Returns ``(interceptor, backend, project_root, event_buffer)``. Callers
    are responsible for tearing the stack down via :func:`_close_stack`.
    """
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    backend = SqliteTaskBackend(cfg)
    await backend.start()

    unique = uuid.uuid4().hex[:8]
    event_buffer = EventBuffer(
        db_path=tmp_path / f'events_{unique}.db', buffer_size_threshold=100,
    )
    await event_buffer.initialize()

    interceptor_config = None
    if enforce:
        interceptor_config = FusedMemoryConfig()
        interceptor_config.task_status.enforce_transitions = True

    interceptor = TaskInterceptor(backend, None, event_buffer, config=interceptor_config)
    return interceptor, backend, str(tmp_path), event_buffer


async def _close_stack(interceptor, backend, event_buffer) -> None:
    """Tear down a stack opened by :func:`_fresh_stack`."""
    await interceptor.close()
    await backend.close()
    await event_buffer.close()


@pytest_asyncio.fixture
async def backend(tmp_path):
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    yield b
    await b.close()


@pytest_asyncio.fixture
async def project_root(tmp_path):
    return str(tmp_path / 'proj')


def test_taskstatus_vocabulary_includes_a_row_status_literals() -> None:
    """Not a boundary cell itself — ties the status string literals A1-A6
    assert against ('pending', 'in-progress', 'infra-hold', 'done') to the
    real ``shared.task_statuses.TaskStatus`` vocabulary, so a future rename
    of a member is caught here instead of leaving this module's asserts
    silently comparing against a stale string.
    """
    values = {s.value for s in TaskStatus}
    for status in ('pending', 'in-progress', 'infra-hold', 'done'):
        assert status in values, (
            f'{status!r} is asserted on throughout this module but is not a '
            f'TaskStatus member: {sorted(values)}'
        )


# ── A1-A4: vocabulary rejection + transition-legality (log/enforce) ────


@pytest.mark.asyncio
@pytest.mark.parametrize('enforce', [False, True])
async def test_a1_vocab_rejection_raises_and_read_path_unaffected(tmp_path, enforce):
    """A1: an out-of-vocabulary status ('in_progress' typo, underscore not
    hyphen) raises TaskmasterError(TASKMASTER_TOOL_ERROR) through the real
    interceptor+backend stack in BOTH log-mode and enforce-mode — this is
    rho1a's vocabulary gate, which sits ahead of (and is orthogonal to) the
    transition-legality gate (a ValueError from TaskStatus coercion inside
    is_legal_transition is treated as legal — never bricking on bad
    vocabulary — so the write reaches the backend, which is the actual
    vocabulary authority). The read path proves the write never landed:
    get_task still shows 'pending' and get_statuses still lists the task —
    still dispatchable.
    """
    interceptor, backend, project_root, event_buffer = await _fresh_stack(tmp_path, enforce=enforce)
    try:
        await backend.add_task(project_root=project_root, title='t')

        with pytest.raises(TaskmasterError) as exc_info:
            await interceptor.set_task_status(
                '1', 'in_progress', project_root, agent_id='orchestrator-x',
            )
        assert exc_info.value.code == 'TASKMASTER_TOOL_ERROR'

        task = await backend.get_task('1', project_root=project_root)
        assert task['status'] == 'pending'

        statuses = await backend.get_statuses(project_root)
        assert statuses['1'] == 'pending'
    finally:
        await _close_stack(interceptor, backend, event_buffer)


@pytest.mark.asyncio
@pytest.mark.parametrize('enforce', [False, True])
async def test_a2_legal_dispatch_no_warning_persists_both_modes(tmp_path, enforce, caplog):
    """A2: pending->in-progress by an ORCHESTRATOR actor is the ordinary
    dispatch transition — legal in BOTH log-mode and enforce-mode, so no
    'illegal_transition would-reject' WARNING fires either way, and the
    write persists (get_statuses reflects 'in-progress') in both modes.
    """
    interceptor, backend, project_root, event_buffer = await _fresh_stack(tmp_path, enforce=enforce)
    try:
        await backend.add_task(project_root=project_root, title='t')

        with caplog.at_level(logging.WARNING, logger='fused_memory.middleware.task_interceptor'):
            result = await interceptor.set_task_status(
                '1', 'in-progress', project_root, agent_id='orchestrator-x',
            )
        assert 'error' not in result, result
        assert not any(
            'illegal_transition would-reject' in r.message for r in caplog.records
        ), [r.message for r in caplog.records]

        statuses = await backend.get_statuses(project_root)
        assert statuses['1'] == 'in-progress'
    finally:
        await _close_stack(interceptor, backend, event_buffer)


@pytest.mark.asyncio
async def test_a3_reconciliation_illegal_transition_log_mode_warns_and_proceeds(tmp_path, caplog):
    """A3 log-mode: RECONCILIATION is the one actor-specific restriction
    (Table A) — in-progress->pending is illegal FOR IT ALONE. In log-mode
    (enforce=False, the default) the gate logs a grep-stable WARNING but the
    write still proceeds: get_statuses shows 'pending' after the call.

    The WARNING assertion pins only the invariant token
    ('illegal_transition would-reject'), not the full sentence (transition/
    actor suffix) — so the test survives a benign log-message rewording
    instead of coupling to free-text prose (the log call has no structured
    ``extra=`` field to assert on instead; see task_interceptor.py:1049-1054).
    """
    interceptor, backend, project_root, event_buffer = await _fresh_stack(tmp_path, enforce=False)
    try:
        await backend.add_task(project_root=project_root, title='t')
        await backend.set_task_status('1', 'in-progress', project_root=project_root)

        with caplog.at_level(logging.WARNING, logger='fused_memory.middleware.task_interceptor'):
            result = await interceptor.set_task_status(
                '1', 'pending', project_root, agent_id='recon-stage-7',
            )
        assert 'error' not in result, result
        assert any(
            'illegal_transition would-reject' in r.message for r in caplog.records
        ), [r.message for r in caplog.records]

        statuses = await backend.get_statuses(project_root)
        assert statuses['1'] == 'pending'
    finally:
        await _close_stack(interceptor, backend, event_buffer)


@pytest.mark.asyncio
async def test_a3_reconciliation_illegal_transition_enforce_mode_rejects(tmp_path):
    """A3 enforce-mode: the same in-progress->pending write by RECONCILIATION
    is typed-rejected (a structured 'illegal_transition' error) and the
    backend row is left unchanged (still 'in-progress') — the write never
    lands.
    """
    interceptor, backend, project_root, event_buffer = await _fresh_stack(tmp_path, enforce=True)
    try:
        await backend.add_task(project_root=project_root, title='t')
        await backend.set_task_status('1', 'in-progress', project_root=project_root)

        result = await interceptor.set_task_status(
            '1', 'pending', project_root, agent_id='recon-stage-7',
        )
        assert result.get('error') == 'illegal_transition', result

        statuses = await backend.get_statuses(project_root)
        assert statuses['1'] == 'in-progress'
    finally:
        await _close_stack(interceptor, backend, event_buffer)


@pytest.mark.asyncio
@pytest.mark.parametrize('agent_id', [None, 'claude-task-9'])
async def test_a4_unknown_or_human_actor_safe_open_in_enforce_mode(tmp_path, agent_id):
    """A4: an UNKNOWN (agent_id=None, header-less) or interactive HUMAN
    (agent_id='claude-task-9') actor gets the safe-open union (D5) — the
    SAME in-progress->pending write RECONCILIATION is denied for (A3) is
    accepted here even in enforce-mode, and persists.
    """
    interceptor, backend, project_root, event_buffer = await _fresh_stack(tmp_path, enforce=True)
    try:
        await backend.add_task(project_root=project_root, title='t')
        await backend.set_task_status('1', 'in-progress', project_root=project_root)

        result = await interceptor.set_task_status(
            '1', 'pending', project_root, agent_id=agent_id,
        )
        assert 'error' not in result, result

        statuses = await backend.get_statuses(project_root)
        assert statuses['1'] == 'pending'
    finally:
        await _close_stack(interceptor, backend, event_buffer)


# ── A5-A6: claimant round-trip + is_stranded over a backend-fetched row ─


@pytest.mark.asyncio
async def test_a5_claimant_round_trip_via_set_task_status_and_release(backend, project_root):
    """A5: the dispatch-time claimant stamp via set_task_status(claimant_run_id=,
    heartbeat_at=) round-trips through get_task (both stamped, status
    in-progress); release via set_task_claimant(None, None) clears both to
    NULL on the same backend-fetched row without touching status.
    """
    await backend.add_task(project_root=project_root, title='t')
    claimant_run_id = compose_claimant_run_id('run', 'sess', os.getpid())
    heartbeat_at = datetime.now(UTC).isoformat()

    await backend.set_task_status(
        '1', 'in-progress', project_root=project_root,
        claimant_run_id=claimant_run_id, heartbeat_at=heartbeat_at,
    )

    task = await backend.get_task('1', project_root=project_root)
    assert task['status'] == 'in-progress'
    assert task['claimant_run_id'] == claimant_run_id
    assert task['heartbeat_at'] == heartbeat_at

    await backend.set_task_claimant(
        '1', project_root=project_root, claimant_run_id=None, heartbeat_at=None,
    )

    released = await backend.get_task('1', project_root=project_root)
    assert released['claimant_run_id'] is None
    assert released['heartbeat_at'] is None
    assert released['status'] == 'in-progress'


@pytest.mark.asyncio
async def test_a5_claimant_atomic_clear_via_set_task_status_done(backend, project_root):
    """A5 atomic-clear variant: set_task_status('done', claimant_run_id=None,
    heartbeat_at=None) releases the claimant in the SAME call that
    terminates the task.
    """
    await backend.add_task(project_root=project_root, title='t')
    claimant_run_id = compose_claimant_run_id('run', 'sess', os.getpid())
    heartbeat_at = datetime.now(UTC).isoformat()
    await backend.set_task_status(
        '1', 'in-progress', project_root=project_root,
        claimant_run_id=claimant_run_id, heartbeat_at=heartbeat_at,
    )

    await backend.set_task_status(
        '1', 'done', project_root=project_root,
        claimant_run_id=None, heartbeat_at=None,
    )

    task = await backend.get_task('1', project_root=project_root)
    assert task['status'] == 'done'
    assert task['claimant_run_id'] is None
    assert task['heartbeat_at'] is None


@pytest.mark.asyncio
async def test_a6_is_stranded_over_backend_fetched_row_stale_then_refreshed(backend, project_root):
    """A6: is_stranded is fed a task dict FETCHED from the real backend
    (get_task) — not a hand-built dict — proving the backend row shape
    feeds the predicate's production surface. A stale heartbeat is
    stranded; refreshing the heartbeat makes the SAME re-fetched row not
    stranded.
    """
    await backend.add_task(project_root=project_root, title='t')
    now = datetime.now(UTC)
    ttl = timedelta(minutes=5)
    stale_heartbeat = (now - timedelta(minutes=10)).isoformat()

    await backend.set_task_status(
        '1', 'in-progress', project_root=project_root,
        claimant_run_id='run-x', heartbeat_at=stale_heartbeat,
    )
    task = await backend.get_task('1', project_root=project_root)
    assert is_stranded(task, now, ttl) is True

    await backend.set_task_claimant(
        '1', project_root=project_root, heartbeat_at=now.isoformat(),
    )
    refreshed = await backend.get_task('1', project_root=project_root)
    assert is_stranded(refreshed, now, ttl) is False


@pytest.mark.asyncio
async def test_a6_is_stranded_false_for_infra_hold_status(backend, project_root):
    """A6 contrast: a first-class 'infra-hold' status is never stranded (the
    in-progress gate alone excludes it), fetched straight from the backend.
    """
    await backend.add_task(project_root=project_root, title='t')
    await backend.set_task_status('1', 'infra-hold', project_root=project_root)

    now = datetime.now(UTC)
    task = await backend.get_task('1', project_root=project_root)
    assert is_stranded(task, now, timedelta(minutes=5)) is False


@pytest.mark.asyncio
async def test_a6_is_stranded_false_for_legacy_metadata_infra_hold_overload(backend, project_root):
    """A6 contrast: the legacy in-progress + metadata.infra_hold=True
    overload (pre-omega4 migration window) is also never stranded, fetched
    straight from the real backend row.
    """
    await backend.add_task(
        project_root=project_root, title='t',
        metadata=json.dumps({'infra_hold': True}),
    )
    await backend.set_task_status('1', 'in-progress', project_root=project_root)

    now = datetime.now(UTC)
    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata'].get('infra_hold') is True  # sanity: round-tripped
    assert is_stranded(task, now, timedelta(minutes=5)) is False
