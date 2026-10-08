"""Atomicity regressions for SqliteTaskBackend's per-project WRITE connection.

A sibling of ``test_sqlite_task_backend.py`` (8,000+ lines), following the
``test_sqlite_task_backend_crash.py`` precedent.

CONNECTION INVENTORY.  The backend opens three kinds of connection per project:

* the cached WRITE connection — legacy isolation, shared by every coroutine, the
  one that commits and rolls back.  It carries both hazards of
  ``plans/recon-sqlite-database-locked-rca-2026-09-16.md`` (a pinned read and a
  connection-wide rollback) and is the one these arms cover;
* the cached autocommit READ connection — never writes and never rolls back
  anyone's write, and is already serialised by ``_read_lock`` /
  ``_fresh_read_conn`` (task 2694's pin-healing guard).  Out of scope;
* ``get_statuses_fresh``'s per-call connection — never shared.  Out of scope.

MEASURED BASELINES, taken on this task's base commit (2026-10-08):

    checkpoint_all vs. add_task          20/20 'database table is locked'  ->  0/20
    cancelled unit vs. add_task          survivor lands (GREEN pre-fix)    ->  survives

The second arm passes on the unfixed code because the per-project write lock
already serialises write units.  It is kept deliberately, as the regression guard
that replacing that lock with ``AtomicConnection.write()`` preserves rollback
isolation.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend

CHECKPOINT_ITERATIONS = 20
TIMEOUT_SECONDS = 10


@pytest_asyncio.fixture
async def backend():
    store = SqliteTaskBackend()
    await store.start()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
def project_root(tmp_path) -> str:
    return str(tmp_path / 'proj')


@pytest.mark.asyncio
async def test_checkpoint_all_and_writes_do_not_lock_each_other_out(backend, project_root):
    """``checkpoint_all`` swallows a failure per root into ``busy == -1`` and a WARNING.

    Measured before the fix, 20/20 iterations logged
    ``checkpoint failed for <root>: database table is locked``.
    """
    await backend.add_task(project_root, title='seed', description='d')

    for i in range(CHECKPOINT_ITERATIONS):
        added, checkpointed = await asyncio.wait_for(
            asyncio.gather(
                backend.add_task(project_root, title=f't{i}', description='d'),
                backend.checkpoint_all(),
                return_exceptions=True,
            ),
            TIMEOUT_SECONDS,
        )
        assert not isinstance(added, BaseException), (
            f'iteration {i}: add_task raised alongside a concurrent checkpoint: {added!r}'
        )
        assert isinstance(checkpointed, dict), checkpointed
        failed = {root: r for root, r in checkpointed.items() if r['busy'] == -1}
        assert not failed, (
            f'iteration {i}: the checkpoint failed alongside a concurrent add_task: {failed}'
        )


@pytest.mark.asyncio
async def test_a_cancelled_write_unit_does_not_discard_a_concurrent_add_task(
    backend, project_root
):
    """Cancelling one write unit must not roll back a concurrent ``add_task``.

    GREEN before the fix: the per-project write lock already serialised write
    units.  Kept as the guard that ``AtomicConnection.write()`` keeps the same
    rollback isolation once it replaces that lock.
    """
    for i in range(1, 4):
        await backend.add_task(project_root, title=f'seed {i}', description='d')

    victim = asyncio.create_task(backend.remove_tasks(['1'], project_root))
    survivor = asyncio.create_task(
        backend.add_task(project_root, title='survivor', description='d')
    )
    await asyncio.sleep(0)
    victim.cancel()

    _, added = await asyncio.wait_for(
        asyncio.gather(victim, survivor, return_exceptions=True), TIMEOUT_SECONDS
    )

    assert isinstance(added, dict), added
    survivor_row = await backend.get_task(added['id'], project_root)
    assert survivor_row['title'] == 'survivor', (
        "the concurrent add_task is gone: the cancelled unit's rollback discarded it"
    )
    assert (await backend.get_task('1', project_root))['id'] == 1, (
        'the cancelled remove_tasks unit deleted its row anyway'
    )
