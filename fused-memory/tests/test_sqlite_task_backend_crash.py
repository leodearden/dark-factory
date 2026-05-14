"""Crash-durability test for :class:`SqliteTaskBackend`.

Regression guard for the 2026-05-13 incident where ~150 task rows committed
over 60+ hours were silently lost when the prior fused-memory instance was
watchdog-SIGABRT'd: a new instance opened the WAL but did not recover the
committed-but-not-checkpointed frames.

A subprocess (``_sqlite_chaos_child.py``) opens the production backend,
writes N task rows via the real ``add_task`` path (each row goes through
``_txn`` and returns only after ``conn.commit()``), journals each returned
id to a separate fsync'd file, and then SIGKILLs itself. The parent then
opens a fresh ``SqliteTaskBackend`` and asserts every journaled id is
present in the DB.

The fundamental durability contract under test:

  *"If ``add_task`` returned, the row must survive an immediate SIGKILL of
   the writer process."*

Pre-Phase-1 (``synchronous=NORMAL`` + default ``wal_autocheckpoint=1000`` +
no explicit checkpoints) this contract is fragile under the prod conditions
captured in the forensic write-up: reader-pinned WAL, multi-day uncheckpointed
window, watchdog SIGABRT mid-write. The kernel page-cache survives SIGKILL
in this test so the failure mode is harder to trigger here than in prod,
but the test still catches any aiosqlite worker-thread / WAL-recovery
regression on the last-commit boundary.

Post-Phase-1 (``synchronous=FULL`` + ``wal_autocheckpoint=100`` + periodic
explicit ``wal_checkpoint(TRUNCATE)``) the contract should hold rock-solid
across many iterations.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_asyncio

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig

_CHILD_SCRIPT = Path(__file__).parent / '_sqlite_chaos_child.py'


def _spawn_chaos_child(project_root: Path, n_commits: int, journal: Path) -> int:
    """Run the chaos child to completion (i.e. self-SIGKILL).

    Returns the child's exit signal so the test can assert it was indeed
    killed via SIGKILL and not a clean exit (which would hide regressions
    where the child errors out before committing anything).
    """
    env = dict(os.environ)
    # Make sure the child sees the in-tree fused_memory src.
    proc = subprocess.run(
        [sys.executable, str(_CHILD_SCRIPT), str(project_root), str(n_commits), str(journal)],
        env=env,
        capture_output=True,
        check=False,
    )
    # The child SIGKILLs itself unconditionally — exitcode should be -9.
    return proc.returncode


@pytest_asyncio.fixture
async def reopened_backend(tmp_path: Path):
    """Open a fresh ``SqliteTaskBackend`` against ``tmp_path`` for the parent
    test's post-kill verification. Yielded only after the chaos child has
    exited so the new backend really is reopening the on-disk state.
    """
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    try:
        yield b
    finally:
        await b.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize('n_commits', [50])
async def test_committed_rows_survive_sigkill(
    tmp_path: Path,
    reopened_backend: SqliteTaskBackend,
    n_commits: int,
) -> None:
    """SIGKILL the writer after N committed add_task calls — verify every
    journaled id survives in the reopened DB."""
    journal_path = tmp_path / 'commit_journal.txt'

    rc = _spawn_chaos_child(tmp_path, n_commits, journal_path)
    # SIGKILL → exitcode -9 in subprocess return semantics.
    assert rc == -9, f'chaos child exited {rc}; expected -9 (SIGKILL)'

    assert journal_path.exists(), 'child journal missing — child died before committing anything'
    committed_ids = [line.strip() for line in journal_path.read_text().splitlines() if line.strip()]
    assert len(committed_ids) == n_commits, (
        f'child journaled {len(committed_ids)} commits; expected {n_commits}'
    )

    listing = await reopened_backend.get_tasks(project_root=str(tmp_path))
    surviving_ids = {str(t['id']) for t in listing['tasks']}

    missing = [tid for tid in committed_ids if tid not in surviving_ids]
    assert not missing, (
        f'{len(missing)} of {len(committed_ids)} committed rows missing '
        f'after SIGKILL+reopen: {missing[:10]}{"..." if len(missing) > 10 else ""}'
    )


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_repeated_sigkill_cycles_preserve_durability(
    tmp_path: Path,
) -> None:
    """Five SIGKILL/reopen cycles back-to-back — each cycle writes 20 rows,
    SIGKILLs, then a fresh backend in this test process verifies survival
    before the next cycle. Catches any cumulative WAL-state corruption that
    only manifests after multiple unclean closes."""
    cycles = 5
    per_cycle = 20
    journal_path = tmp_path / 'commit_journal.txt'

    for cycle in range(cycles):
        rc = _spawn_chaos_child(tmp_path, per_cycle, journal_path)
        assert rc == -9, f'cycle {cycle}: child exited {rc}'

        # Verify in-process by opening a fresh backend each cycle.
        cfg = TaskmasterConfig(project_root=str(tmp_path))
        b = SqliteTaskBackend(cfg)
        await b.start()
        try:
            listing = await b.get_tasks(project_root=str(tmp_path))
        finally:
            await b.close()

        surviving_ids = {str(t['id']) for t in listing['tasks']}
        expected_ids = {
            line.strip() for line in journal_path.read_text().splitlines() if line.strip()
        }
        missing = expected_ids - surviving_ids
        assert not missing, (
            f'cycle {cycle}: {len(missing)} committed rows lost across '
            f'{cycle + 1} SIGKILL cycles'
        )
