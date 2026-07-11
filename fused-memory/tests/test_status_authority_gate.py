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

import uuid

import pytest_asyncio

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.backends.task_backend_errors import TaskmasterError
from fused_memory.config.schema import FusedMemoryConfig, TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer
from shared.task_claimant import compose_claimant_run_id, is_stranded
from shared.task_statuses import TaskStatus

__all__ = [
    'TaskmasterError',
    'TaskStatus',
    'compose_claimant_run_id',
    'is_stranded',
]


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
