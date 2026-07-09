"""Boundary/integration gate for the candidate_key seam.

fm-task-dedup W8 task A3 (PRD ``plans/fm-task-dedup-prd.md`` §8 + §6b
BT-A1…BT-A5). Drives the REAL production paths end-to-end: a real
:class:`SqliteTaskBackend` (with its partial UNIQUE index on
``(tag, candidate_key)``) behind a real :class:`TaskInterceptor`, via both
the ``submit_task``→``resolve_ticket`` ticket path and the synchronous
``planning_mode`` path — plus crash-injection (BT-A3) and cross-restart
(BT-A4) cases that have no existing coverage.

Complements the existing isolated-unit tests rather than duplicating them:

* Backend-only producer-side: ``test_sqlite_task_backend.py`` (~2500-2606).
* Interceptor/planning_mode consumer-side with a MOCKED ``add_task``:
  ``test_task_interceptor.py`` (~5935, ~7838).

This suite is the INTEGRATION gate: the real interceptor + real backend +
real index wired together so producer-raise and consumer-combine fire in
ONE real flow.

Every submission here uses plain ``metadata={'files': [...]}`` (no
``escalation_id``/``suggestion_hash``) so the only dedup mechanism in play
is the durable candidate_key index — and the curator is disabled
(``config=None``) so every ticket dispatches straight to ``tm.add_task``,
faithfully reproducing the "in-memory dedup missed it" scenario the durable
index exists to catch.
"""

from __future__ import annotations

import json
import uuid

import pytest
import pytest_asyncio
from _fm_helpers import submit_and_resolve

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.backends.task_backend_errors import DuplicateCandidateKeyError
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer


async def open_fresh_stack(tmp_path):
    """Open a brand-new (interceptor, backend, project_root, event_buffer)
    stack rooted at *tmp_path*, with the curator OFF (``config=None``) so
    ``TaskInterceptor._get_curator()`` short-circuits to ``None`` and every
    ticket dispatches CREATE straight through to ``tm.add_task`` — both
    identical submissions reach the backend's partial UNIQUE index instead
    of being deduped in-memory by a curator.

    Each call gets its own uniquely-named ticket-store/event-buffer files
    (even across repeated calls against the SAME ``tmp_path``) so a second
    call — as used by the BT-A4 restart test — is a genuinely cold
    interceptor instance, not one that inherits the prior stack's ticket
    history. The backend, by contrast, always opens the SAME on-disk
    ``tasks.db`` under ``tmp_path`` — that persistence (and its partial
    UNIQUE index) is exactly the property BT-A3/BT-A4 exercise.

    Returns ``(interceptor, backend, project_root, event_buffer)``. Callers
    are responsible for tearing the stack down via :func:`close_stack`.
    """
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    backend = SqliteTaskBackend(cfg)
    await backend.start()

    unique = uuid.uuid4().hex[:8]
    event_buffer = EventBuffer(
        db_path=tmp_path / f'events_{unique}.db', buffer_size_threshold=100,
    )
    await event_buffer.initialize()
    store = TicketStore(tmp_path / f'tickets_{unique}.db')
    await store.initialize()

    interceptor = TaskInterceptor(backend, None, event_buffer, config=None, ticket_store=store)
    return interceptor, backend, str(tmp_path), event_buffer


async def close_stack(interceptor, backend, event_buffer) -> None:
    """Tear down a stack opened by :func:`open_fresh_stack`.

    ``interceptor.close()`` cancels any running per-project ticket workers
    and closes the wired-in :class:`TicketStore` internally; this helper
    additionally closes the backend and event buffer, which the
    interceptor does not own.
    """
    await interceptor.close()
    await backend.close()
    await event_buffer.close()


@pytest_asyncio.fixture
async def real_stack(tmp_path):
    """A real ``(interceptor, backend, project_root)`` stack — curator OFF.

    Used by BT-A1/BT-A2/BT-A5, which each need exactly one live stack for
    the duration of the test. BT-A3/BT-A4 manage their own stack(s)
    directly via :func:`open_fresh_stack`/:func:`close_stack` since they
    need to close and reopen mid-test.
    """
    interceptor, backend, project_root, event_buffer = await open_fresh_stack(tmp_path)
    try:
        yield interceptor, backend, project_root
    finally:
        await close_stack(interceptor, backend, event_buffer)


async def count_non_cancelled(backend, project_root: str) -> int:
    """Count non-cancelled rows for *project_root*.

    ``get_tasks`` returns cancelled rows too, so callers that need a
    no-orphan / exactly-one-survivor count must filter client-side.
    """
    listing = await backend.get_tasks(project_root=project_root)
    return len([t for t in listing['tasks'] if t['status'] != 'cancelled'])


# ── BT-A1: curator submit→resolve, two-way integration ──────────────────


@pytest.mark.asyncio
async def test_bt_a1_submit_resolve_two_way_candidate_key_collision(real_stack):
    """Two submit_and_resolve calls with the same normalized (title, files)
    — curator OFF (config=None) so BOTH tickets dispatch CREATE straight to
    tm.add_task. The 1st commits; the 2nd trips the real partial UNIQUE
    index on (tag, candidate_key) (producer/backend), and the interceptor's
    create-dispatch resolves the raised DuplicateCandidateKeyError as
    'combined' (consumer) — the end-to-end integration the isolated unit
    tests (backend-only producer test; interceptor test with a mocked
    add_task) do not exercise together.
    """
    interceptor, backend, project_root = real_stack

    first = await submit_and_resolve(
        interceptor, project_root,
        title='Fix parser',
        metadata={'files': ['a.py', 'b.py']},
    )
    assert first['id'] == '1', f'expected the first submit to create id=1; got {first!r}'

    # Case + extra internal whitespace on the title, file order swapped —
    # compute_candidate_key is case/whitespace-insensitive on title and
    # order-insensitive on files, so this collides with the first row.
    second = await submit_and_resolve(
        interceptor, project_root,
        title='fix  parser',
        metadata={'files': ['b.py', 'a.py']},
    )
    assert second['id'] == first['id'], (
        f'expected the collision to resolve onto id={first["id"]!r}; got {second!r}'
    )
    assert second.get('action') == 'candidate_key_collision', second
    assert second.get('deduplicated') is True, second

    non_cancelled = await count_non_cancelled(backend, project_root)
    assert non_cancelled == 1, (
        f'expected exactly one non-cancelled row (no orphan from the rejected '
        f'2nd insert); got {non_cancelled}'
    )


# ── BT-A2: planning_mode reintroduction guard, two-way ───────────────────


@pytest.mark.asyncio
async def test_bt_a2_planning_mode_reintroduction_guard(real_stack):
    """Two identical planning_mode submit_task calls. planning_mode bypasses
    the ticket store/curator entirely and calls tm.add_task directly with
    status='deferred'; the 2nd call's insert trips the real partial UNIQUE
    index and _submit_task_planning_mode resolves the raised
    DuplicateCandidateKeyError as a combine — closing the planning-mode
    duplicate-reintroduction gap end-to-end (not just against a mocked
    add_task).
    """
    interceptor, backend, project_root = real_stack

    first = await interceptor.submit_task(
        project_root,
        planning_mode=True,
        title='Fix parser',
        metadata={'files': ['a.py', 'b.py']},
    )
    assert first['task_id'] == '1', first
    assert first['status'] == 'deferred', first
    assert first['planning_mode'] is True, first

    # Case + extra internal whitespace on the title, file order swapped —
    # same normalized candidate_key as the first submission.
    second = await interceptor.submit_task(
        project_root,
        planning_mode=True,
        title='fix  parser',
        metadata={'files': ['b.py', 'a.py']},
    )
    assert second.get('combined') is True, second
    assert second.get('planning_mode') is True, second
    assert second.get('task_id') == first['task_id'], (
        f'expected the collision to name survivor task_id={first["task_id"]!r}; got {second!r}'
    )

    non_cancelled = await count_non_cancelled(backend, project_root)
    assert non_cancelled == 1, (
        f'expected exactly one non-cancelled (deferred) row (no orphan); got {non_cancelled}'
    )


# ── BT-A3: crash injected between INSERT and COMMIT ──────────────────────


@pytest.mark.asyncio
async def test_bt_a3_crash_between_insert_and_commit_leaves_no_orphan(tmp_path):
    """A crash injected between the tasks INSERT and the txn commit must
    roll back cleanly (zero orphan rows), and the on-disk partial UNIQUE
    index must still be intact both immediately after and across a
    reconnect.

    RED until step-4 adds the ``_after_insert_fault_hook`` seam: without
    it, add_task never invokes the hook, the INSERT commits, no exception
    is raised, and the assertions below fail.
    """
    project_root = str(tmp_path)
    cfg = TaskmasterConfig(project_root=project_root)
    backend = SqliteTaskBackend(cfg)
    await backend.start()
    try:
        before = await count_non_cancelled(backend, project_root)
        assert before == 0

        def _boom():
            raise RuntimeError('injected crash between INSERT and COMMIT')

        backend._after_insert_fault_hook = _boom
        try:
            with pytest.raises(RuntimeError, match='injected crash between INSERT and COMMIT'):
                await backend.add_task(
                    project_root=project_root,
                    title='Fix parser',
                    metadata=json.dumps({'files': ['a.py', 'b.py']}),
                )
        finally:
            # Clear the hook unconditionally — even if add_task raised
            # something other than the expected RuntimeError (e.g. a
            # pytest.raises mismatch), the hook must not stay armed for
            # the post-crash add_task calls below.
            backend._after_insert_fault_hook = None

        after_crash = await count_non_cancelled(backend, project_root)
        assert after_crash == before, (
            f'expected the failed INSERT to roll back with zero orphan rows; '
            f'before={before}, after crash={after_crash}'
        )

        # No phantom row blocks a normal add — the rolled-back insert left no trace.
        created = await backend.add_task(
            project_root=project_root,
            title='Fix parser',
            metadata=json.dumps({'files': ['a.py', 'b.py']}),
        )
        assert created['id'] == '1', created

        # The index invariant is intact — an immediate duplicate still collides.
        with pytest.raises(DuplicateCandidateKeyError) as exc_info:
            await backend.add_task(
                project_root=project_root,
                title='fix  parser',
                metadata=json.dumps({'files': ['b.py', 'a.py']}),
            )
        assert exc_info.value.existing_id == 1, exc_info.value.existing_id
    finally:
        await backend.close()

    # Reconnect: the on-disk partial UNIQUE index survives across a fresh
    # backend instance on the same tmp_path.
    reopened = SqliteTaskBackend(cfg)
    await reopened.start()
    try:
        with pytest.raises(DuplicateCandidateKeyError) as exc_info:
            await reopened.add_task(
                project_root=project_root,
                title='Fix parser',
                metadata=json.dumps({'files': ['a.py', 'b.py']}),
            )
        assert exc_info.value.existing_id == 1, exc_info.value.existing_id
    finally:
        await reopened.close()


# ── BT-A4: cross-restart durability ───────────────────────────────────────


@pytest.mark.asyncio
async def test_bt_a4_restart_durability_combine_still_fires(tmp_path):
    """The candidate_key collision still combines after closing the whole
    stack and opening a FRESH backend + interceptor (cold in-memory caches)
    on the SAME tmp_path — proving the guarantee is a durable DB-index
    property, not an artefact of the six in-memory dedup layers (which are
    all cold after a restart).
    """
    project_root = str(tmp_path)

    interceptor1, backend1, _pr1, event_buffer1 = await open_fresh_stack(tmp_path)
    created = await submit_and_resolve(
        interceptor1, project_root,
        title='Fix parser',
        metadata={'files': ['a.py', 'b.py']},
    )
    assert created['id'] == '1', created
    await close_stack(interceptor1, backend1, event_buffer1)

    # Phase 2: fresh stack, cold caches, same on-disk tasks.db + index.
    interceptor2, backend2, _pr2, event_buffer2 = await open_fresh_stack(tmp_path)
    try:
        combined = await submit_and_resolve(
            interceptor2, project_root,
            title='fix  parser',
            metadata={'files': ['b.py', 'a.py']},
        )
        assert combined['id'] == created['id'], (
            f'expected the fresh stack to still combine onto id={created["id"]!r}; '
            f'got {combined!r}'
        )
        assert combined.get('action') == 'candidate_key_collision', combined
        assert combined.get('deduplicated') is True, combined

        # Direct fresh-backend producer-side check too — durable independent
        # of the interceptor's own (also-cold) in-memory dedup layers.
        with pytest.raises(DuplicateCandidateKeyError) as exc_info:
            await backend2.add_task(
                project_root=project_root,
                title='Fix parser',
                metadata=json.dumps({'files': ['a.py', 'b.py']}),
            )
        assert str(exc_info.value.existing_id) == created['id'], exc_info.value.existing_id

        non_cancelled = await count_non_cancelled(backend2, project_root)
        assert non_cancelled == 1, non_cancelled
    finally:
        await close_stack(interceptor2, backend2, event_buffer2)


# ── BT-A5: cancel + re-file → new row, no false combine ─────────────────


@pytest.mark.asyncio
async def test_bt_a5_cancel_then_refile_creates_new_row(real_stack):
    """Cancelling the survivor then re-filing the identical (title, files)
    via the real interceptor path creates a NEW non-cancelled row — the
    partial UNIQUE index excludes cancelled rows
    (``WHERE ... status != 'cancelled'``), so this is a legitimate refile,
    not a false combine. Complements the backend-only BT-A5 unit
    (test_sqlite_task_backend.py) by driving the real interceptor.
    """
    interceptor, backend, project_root = real_stack

    first = await submit_and_resolve(
        interceptor, project_root,
        title='Fix parser',
        metadata={'files': ['a.py', 'b.py']},
    )
    assert first['id'] == '1', first

    await backend.set_task_status(first['id'], 'cancelled', project_root=project_root)

    second = await submit_and_resolve(
        interceptor, project_root,
        title='Fix parser',
        metadata={'files': ['a.py', 'b.py']},
    )
    assert second['id'] == '2', (
        f'expected the refile to land as a new id=2 (cancelled rows are '
        f'excluded from the partial index); got {second!r}'
    )
    assert not second.get('deduplicated'), (
        f'expected a create disposition, not a combine; got {second!r}'
    )
    assert second.get('action') != 'candidate_key_collision', second

    listing = await backend.get_tasks(project_root=project_root)
    statuses_by_id = {t['id']: t['status'] for t in listing['tasks']}
    assert statuses_by_id == {'1': 'cancelled', '2': 'pending'}, statuses_by_id
