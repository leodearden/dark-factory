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
