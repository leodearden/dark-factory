"""Tests for SQLite pragma triad applied across orchestrator stores.

Mirrors the assertion style from shared/tests/test_async_sqlite_base.py::TestApplyFullDurabilityPragmas.
Each store is tested for:
  1. Full pragma triad on init (journal_mode='wal', busy_timeout=5000, synchronous=2,
     wal_autocheckpoint=100, journal_size_limit=67108864).
  2. checkpoint() returns CheckpointResult with busy==0.
  3. checkpoint() truncates the WAL (≤32 bytes) after writes.
  4. checkpoint() is idempotent.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from orchestrator.harness import TaskReport
from orchestrator.run_store import RunStore
from orchestrator.workflow import WorkflowOutcome
from shared.sqlite_sync_base import CheckpointResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_PRAGMAS = {
    'journal_mode': 'wal',
    'busy_timeout': 5000,
    'synchronous': 2,  # 2 == FULL
    'wal_autocheckpoint': 100,
    'journal_size_limit': 67108864,
}


def _assert_pragma_triad(store: object) -> None:
    """Open a fresh connection via the store's _connect() and assert all five pragmas.

    Per-connection pragmas (busy_timeout, synchronous, wal_autocheckpoint,
    journal_size_limit) are not persisted in the DB file — they must be verified
    on the connection that set them.  We use store._connect() because that is
    exactly what every public method uses, and testing it directly confirms that
    the full triad is applied on every new connection the store opens.

    journal_mode persists in the DB file header and is also verified here for
    completeness.
    """
    conn = store._connect()  # type: ignore[attr-defined]
    try:
        assert conn.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
        assert conn.execute('PRAGMA busy_timeout').fetchone()[0] == 5000
        assert conn.execute('PRAGMA synchronous').fetchone()[0] == 2
        assert conn.execute('PRAGMA wal_autocheckpoint').fetchone()[0] == 100
        assert conn.execute('PRAGMA journal_size_limit').fetchone()[0] == 67108864
    finally:
        conn.close()


def _make_task_report(task_id: str = 't1') -> TaskReport:
    return TaskReport(
        task_id=task_id,
        title='Test task',
        outcome=WorkflowOutcome.DONE,
        cost_usd=0.10,
        duration_ms=1000,
        agent_invocations=1,
        execute_iterations=1,
        verify_attempts=0,
        review_cycles=0,
        steward_cost_usd=0.0,
        steward_invocations=0,
        completed_at='2026-01-01T00:00:00+00:00',
    )


# ---------------------------------------------------------------------------
# TestRunStorePragmas
# ---------------------------------------------------------------------------


class TestRunStorePragmas:
    """RunStore applies the full pragma triad to every connection."""

    def test_init_applies_full_pragma_triad(self, tmp_path: Path) -> None:
        """Constructing RunStore initialises WAL mode; _connect() applies full triad."""
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        # _assert_pragma_triad uses store._connect() to verify all five pragmas,
        # including per-connection settings that do not persist in the DB file.
        _assert_pragma_triad(store)

    def test_save_task_result_connection_applies_triad(self, tmp_path: Path) -> None:
        """After save_task_result(), a fresh _connect() still applies the full triad.

        Verifies that _connect() consistently applies all five pragmas on every
        new connection the store opens, not just during _ensure_schema().
        """
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        store.start_run('run-1', 'proj', '2026-01-01T00:00:00+00:00')
        store.save_task_result('run-1', _make_task_report('t1'), 'proj')

        # Verify that a fresh connection via _connect() has all five pragmas applied.
        _assert_pragma_triad(store)

    def test_checkpoint_returns_result_with_busy_zero(self, tmp_path: Path) -> None:
        """checkpoint() on a fresh store returns CheckpointResult with busy==0."""
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        result = store.checkpoint()
        assert isinstance(result, CheckpointResult)
        assert result.busy == 0

    def test_checkpoint_truncates_wal_after_writes(self, tmp_path: Path) -> None:
        """After writes, checkpoint() truncates the WAL file to ≤32 bytes.

        The stores use a connect-per-call pattern: each operation opens and closes
        a connection, which triggers SQLite's PASSIVE close-checkpoint.  To reliably
        accumulate WAL frames before calling checkpoint(), we use a long-lived
        raw connection that stays open across all writes (preventing close-checkpoint
        from firing between writes), then verify WAL size *before* closing it.
        """
        db_path = tmp_path / 'runs.db'
        wal_path = Path(str(db_path) + '-wal')

        store = RunStore(db_path)  # creates schema tables

        # Long-lived writer: keeps the connection open across commits so the WAL
        # accumulates without the PASSIVE close-checkpoint truncating it between writes.
        writer = sqlite3.connect(str(db_path))
        try:
            writer.execute('PRAGMA journal_mode=WAL')
            for i in range(10):
                writer.execute(
                    'INSERT OR IGNORE INTO runs (run_id, project_id, started_at) '
                    'VALUES (?, ?, ?)',
                    (f'run-wal-{i}', 'proj', '2026-01-01T00:00:00+00:00'),
                )
            writer.commit()

            # Check WAL while the connection is still open — no close-checkpoint yet.
            before_size = wal_path.stat().st_size if wal_path.exists() else 0
            assert before_size > 0, (
                'Expected WAL frames to exist while writer connection is still open'
            )
        finally:
            writer.close()

        # After writer closes a PASSIVE checkpoint fires (moves frames to DB but
        # does not necessarily truncate the physical WAL file).  Regardless,
        # store.checkpoint() must produce a TRUNCATE result with busy==0 and
        # leave the WAL file at ≤32 bytes.
        result = store.checkpoint()
        assert result.busy == 0
        after_size = wal_path.stat().st_size if wal_path.exists() else 0
        assert after_size <= 32, (
            f'WAL file should be ≤32 bytes after TRUNCATE checkpoint; got {after_size}'
        )

    def test_checkpoint_is_idempotent(self, tmp_path: Path) -> None:
        """Two back-to-back checkpoint() calls both succeed and return CheckpointResult."""
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        store.start_run('run-1', 'proj', '2026-01-01T00:00:00+00:00')

        result1 = store.checkpoint()
        result2 = store.checkpoint()
        assert isinstance(result1, CheckpointResult)
        assert isinstance(result2, CheckpointResult)
        assert result1.busy == 0
        assert result2.busy == 0
