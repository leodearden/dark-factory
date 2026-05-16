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


def _assert_pragma_triad(db_path: Path) -> None:
    """Open a fresh independent connection to db_path and assert all five pragmas."""
    conn = sqlite3.connect(str(db_path))
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
        """Constructing RunStore sets all five pragmas on the DB file."""
        db_path = tmp_path / 'runs.db'
        RunStore(db_path)
        _assert_pragma_triad(db_path)

    def test_save_task_result_connection_applies_triad(self, tmp_path: Path) -> None:
        """After save_task_result(), the DB file still has the full pragma triad.

        Since the store's _connect() applies pragmas on every connection, the
        DB-level settings (journal_mode persists in file header; synchronous,
        wal_autocheckpoint, journal_size_limit are per-connection but we verify
        by opening a fresh connection through the store's DB path).
        """
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        store.start_run('run-1', 'proj', '2026-01-01T00:00:00+00:00')
        store.save_task_result('run-1', _make_task_report('t1'), 'proj')

        # Open a brand-new connection (independent of the store) and verify
        # the DB-persistent pragma (journal_mode) and the file-system state.
        _assert_pragma_triad(db_path)

    def test_checkpoint_returns_result_with_busy_zero(self, tmp_path: Path) -> None:
        """checkpoint() on a fresh store returns CheckpointResult with busy==0."""
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        result = store.checkpoint()
        assert isinstance(result, CheckpointResult)
        assert result.busy == 0

    def test_checkpoint_truncates_wal_after_writes(self, tmp_path: Path) -> None:
        """After writes, checkpoint() truncates the WAL file to ≤32 bytes."""
        db_path = tmp_path / 'runs.db'
        wal_path = Path(str(db_path) + '-wal')

        store = RunStore(db_path)
        # Write some data to grow the WAL
        store.start_run('run-1', 'proj', '2026-01-01T00:00:00+00:00')
        for i in range(5):
            store.save_task_result('run-1', _make_task_report(f't{i}'), 'proj')

        assert wal_path.exists() and wal_path.stat().st_size > 0, (
            'WAL file should have grown after writes'
        )

        result = store.checkpoint()
        assert result.busy == 0
        # After TRUNCATE checkpoint the WAL should be at most the 32-byte SQLite WAL header.
        assert wal_path.stat().st_size <= 32, (
            f'WAL file should be ≤32 bytes after truncate checkpoint, '
            f'got {wal_path.stat().st_size}'
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
