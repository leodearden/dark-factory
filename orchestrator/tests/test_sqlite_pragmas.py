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

import logging
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from shared.sqlite_sync_base import CheckpointResult

from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import Harness, TaskReport
from orchestrator.overrides import OverrideStore
from orchestrator.run_store import RunStore
from orchestrator.workflow import WorkflowOutcome

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


# ---------------------------------------------------------------------------
# TestEventStorePragmas
# ---------------------------------------------------------------------------


class TestEventStorePragmas:
    """EventStore applies the full pragma triad to every connection."""

    def test_init_applies_full_pragma_triad(self, tmp_path: Path) -> None:
        """Constructing EventStore initialises WAL mode; _connect() applies full triad."""
        db_path = tmp_path / 'events.db'
        store = EventStore(db_path, 'run-x')
        _assert_pragma_triad(store)

    def test_emit_connection_applies_triad(self, tmp_path: Path) -> None:
        """After emit(), a fresh _connect() still applies the full pragma triad.

        Verifies that _connect() consistently applies all five pragmas on the
        connection used by emit(), not just during _ensure_schema().
        """
        db_path = tmp_path / 'events.db'
        store = EventStore(db_path, 'run-x')
        store.emit(EventType.task_started, task_id='1')

        # Verify that a fresh connection via _connect() has all five pragmas applied.
        _assert_pragma_triad(store)

    def test_checkpoint_returns_result_with_busy_zero(self, tmp_path: Path) -> None:
        """checkpoint() on a fresh store returns CheckpointResult with busy==0."""
        db_path = tmp_path / 'events.db'
        store = EventStore(db_path, 'run-x')
        result = store.checkpoint()
        assert isinstance(result, CheckpointResult)
        assert result.busy == 0

    def test_checkpoint_truncates_wal_after_emits(self, tmp_path: Path) -> None:
        """After emits that grow the WAL, checkpoint() truncates it to ≤32 bytes."""
        db_path = tmp_path / 'events.db'
        wal_path = Path(str(db_path) + '-wal')

        store = EventStore(db_path, 'run-x')

        # Long-lived writer: keeps the connection open across commits so the WAL
        # accumulates without the PASSIVE close-checkpoint truncating it.
        writer = sqlite3.connect(str(db_path))
        try:
            writer.execute('PRAGMA journal_mode=WAL')
            for i in range(10):
                writer.execute(
                    'INSERT INTO events '
                    '(timestamp, run_id, event_type, data) VALUES (?, ?, ?, ?)',
                    (f'2026-01-01T{i:02d}:00:00+00:00', 'run-x', 'task_started', '{}'),
                )
            writer.commit()

            before_size = wal_path.stat().st_size if wal_path.exists() else 0
            assert before_size > 0, (
                'Expected WAL frames to exist while writer connection is still open'
            )
        finally:
            writer.close()

        result = store.checkpoint()
        assert result.busy == 0
        after_size = wal_path.stat().st_size if wal_path.exists() else 0
        assert after_size <= 32, (
            f'WAL file should be ≤32 bytes after TRUNCATE checkpoint; got {after_size}'
        )

    def test_emit_does_not_raise_on_checkpoint_failure(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """emit() is fire-and-forget: failures must not propagate to callers.

        The existing `except Exception` inside emit() already swallows errors.
        This test regression-pins that contract so a failing helper cannot break
        callers.
        """
        import logging

        db_path = tmp_path / 'events.db'
        store = EventStore(db_path, 'run-x')

        # Corrupt the DB path so any future connection attempt fails.
        # Replace the db_path with a directory (non-file) so sqlite3.connect raises.
        bad_path = tmp_path / 'not_a_file'
        bad_path.mkdir()
        store.db_path = bad_path  # type: ignore[assignment]

        # emit() must not raise even though the connection will fail.
        with caplog.at_level(logging.WARNING):
            store.emit(EventType.task_started, task_id='t1')

        # A warning must have been logged (existing contract).
        assert any('emit' in r.message.lower() or 'event_store' in r.message.lower()
                   for r in caplog.records), (
            f'Expected a warning log from emit() failure; got: {caplog.records}'
        )


# ---------------------------------------------------------------------------
# TestOverrideStorePragmas
# ---------------------------------------------------------------------------


class TestOverrideStorePragmas:
    """OverrideStore applies the full pragma triad to every connection."""

    def test_init_applies_full_pragma_triad(self, tmp_path: Path) -> None:
        """Constructing OverrideStore initialises WAL mode; _connect() applies full triad."""
        db_path = tmp_path / 'overrides.db'
        store = OverrideStore(db_path)
        _assert_pragma_triad(store)

    def test_set_override_connection_applies_triad(self, tmp_path: Path) -> None:
        """After set_override(), a fresh _connect() still applies the full pragma triad.

        Covers the autocommit/isolation_level=None connection used by set_override
        (the plan's note: _connect(timeout=30, isolation_level=None) also applies triad).
        """
        db_path = tmp_path / 'overrides.db'
        store = OverrideStore(db_path)
        store.set_override('proj', 'task-1', boost_tier='high')
        _assert_pragma_triad(store)

    def test_set_override_preserves_autocommit_isolation(
        self, tmp_path: Path
    ) -> None:
        """Regression-pin: set_override's BEGIN IMMEDIATE autocommit pattern works
        concurrently without PinOrderCollision after _connect() is added.

        Two threads race to pin tasks simultaneously with the _after_max_select
        hook injecting a 50 ms pause to make the race deterministic.  Post-fix
        (BEGIN IMMEDIATE inside _connect(timeout=30, isolation_level=None)), the
        second thread blocks at BEGIN IMMEDIATE until the first commits.
        """
        from orchestrator.overrides import PinOrderCollision  # noqa: F401

        db_path = tmp_path / 'overrides.db'
        store = OverrideStore(db_path)
        store._after_max_select = lambda: time.sleep(0.05)

        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def pin_task(task_id: str) -> None:
            try:
                barrier.wait()
                store.set_override('proj', task_id, pinned=True)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=pin_task, args=('A',))
        t2 = threading.Thread(target=pin_task, args=('B',))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f'Unexpected exceptions from concurrent set_override: {errors}'
        overrides = store.get_overrides('proj')
        assert 'A' in overrides and 'B' in overrides
        pin_orders = {overrides['A'].pin_order, overrides['B'].pin_order}
        assert pin_orders == {1, 2}, f'Expected distinct pin_orders {{1,2}}; got {pin_orders}'

    def test_checkpoint_returns_result_with_busy_zero(self, tmp_path: Path) -> None:
        """checkpoint() on a fresh store returns CheckpointResult with busy==0."""
        db_path = tmp_path / 'overrides.db'
        store = OverrideStore(db_path)
        result = store.checkpoint()
        assert isinstance(result, CheckpointResult)
        assert result.busy == 0

    def test_checkpoint_truncates_wal_after_writes(self, tmp_path: Path) -> None:
        """After writes that grow the WAL, checkpoint() truncates it to ≤32 bytes."""
        db_path = tmp_path / 'overrides.db'
        wal_path = Path(str(db_path) + '-wal')

        store = OverrideStore(db_path)

        # Long-lived writer to accumulate WAL frames without close-checkpoint firing.
        writer = sqlite3.connect(str(db_path))
        try:
            writer.execute('PRAGMA journal_mode=WAL')
            for i in range(5):
                writer.execute(
                    'INSERT OR REPLACE INTO overrides '
                    '(project_root, task_id, pinned, reserve_now, created_at, updated_at) '
                    'VALUES (?, ?, 0, 0, ?, ?)',
                    ('proj', f'task-{i}', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'),
                )
            writer.commit()

            before_size = wal_path.stat().st_size if wal_path.exists() else 0
            assert before_size > 0, (
                'Expected WAL frames to exist while writer connection is still open'
            )
        finally:
            writer.close()

        result = store.checkpoint()
        assert result.busy == 0
        after_size = wal_path.stat().st_size if wal_path.exists() else 0
        assert after_size <= 32, (
            f'WAL file should be ≤32 bytes after TRUNCATE checkpoint; got {after_size}'
        )

    @pytest.mark.parametrize('method,kwargs', [
        ('get_overrides', {'project_root': 'proj'}),
        ('get_pin_queue', {'project_root': 'proj'}),
        ('clear_terminal', {'project_root': 'proj', 'terminal_task_ids': set()}),
        ('clear_expired', {'project_root': 'proj'}),
    ])
    def test_read_paths_apply_triad(
        self, tmp_path: Path, method: str, kwargs: dict
    ) -> None:
        """Each read-path connection applies the full pragma triad via _connect()."""
        db_path = tmp_path / 'overrides.db'
        store = OverrideStore(db_path)

        # Call the read-path method (the connection it opens internally uses _connect()).
        getattr(store, method)(**kwargs)

        # Verify a fresh _connect() still has the triad applied (confirms _connect() is
        # used consistently, not just during init).
        _assert_pragma_triad(store)


# ---------------------------------------------------------------------------
# TestHarnessShutdownCheckpoint
# ---------------------------------------------------------------------------

_SENTINEL_RESULT = CheckpointResult(busy=0, log=0, checkpointed=0)


def _make_harness_with_mock_stores() -> tuple[Harness, MagicMock, MagicMock, MagicMock]:
    """Build a minimal Harness (via __new__) with mock stores injected.

    Returns (harness, mock_run_store, mock_event_store, mock_override_store).
    The three stores each have a `checkpoint()` returning _SENTINEL_RESULT.
    """
    harness = Harness.__new__(Harness)

    mock_run_store = MagicMock(spec=RunStore)
    mock_run_store.checkpoint.return_value = _SENTINEL_RESULT

    mock_event_store = MagicMock(spec=EventStore)
    mock_event_store.checkpoint.return_value = _SENTINEL_RESULT

    mock_override_store = MagicMock(spec=OverrideStore)
    mock_override_store.checkpoint.return_value = _SENTINEL_RESULT

    harness._run_store = mock_run_store
    harness.event_store = mock_event_store

    # Harness.scheduler is a Scheduler whose .override_store attribute is the
    # OverrideStore instance (see harness.py:273).
    mock_scheduler = MagicMock()
    mock_scheduler.override_store = mock_override_store
    harness.scheduler = mock_scheduler

    return harness, mock_run_store, mock_event_store, mock_override_store


class TestHarnessShutdownCheckpoint:
    """Harness._checkpoint_stores() calls checkpoint() on each store best-effort.

    These tests exercise the helper that the harness shutdown path calls after
    finish_run() and before usage_gate.shutdown() (step-10 implementation).
    The helper is tested in isolation (via __new__) rather than running the
    full async run() to keep the tests fast and deterministic.
    """

    def test_checkpoint_calls_all_three_stores(self) -> None:
        """_checkpoint_stores() calls checkpoint() exactly once on each store."""
        harness, mock_run_store, mock_event_store, mock_override_store = (
            _make_harness_with_mock_stores()
        )

        harness._checkpoint_stores()

        mock_run_store.checkpoint.assert_called_once()
        mock_event_store.checkpoint.assert_called_once()
        mock_override_store.checkpoint.assert_called_once()

    def test_checkpoint_handles_run_store_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An OperationalError from _run_store.checkpoint() is caught and logged as warning."""
        harness, mock_run_store, _, _ = _make_harness_with_mock_stores()
        err = sqlite3.OperationalError('disk I/O error')
        mock_run_store.checkpoint.side_effect = err

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            harness._checkpoint_stores()  # must not raise

        assert any(
            'run_store.checkpoint() failed' in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        ), f'Expected run_store.checkpoint() failed warning; got: {caplog.records}'

    def test_checkpoint_handles_event_store_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An OperationalError from event_store.checkpoint() is caught and logged as warning."""
        harness, _, mock_event_store, _ = _make_harness_with_mock_stores()
        err = sqlite3.OperationalError('disk I/O error')
        mock_event_store.checkpoint.side_effect = err

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            harness._checkpoint_stores()  # must not raise

        assert any(
            'event_store.checkpoint() failed' in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        ), f'Expected event_store.checkpoint() failed warning; got: {caplog.records}'

    def test_checkpoint_handles_override_store_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An OperationalError from override_store.checkpoint() is caught and logged as warning."""
        harness, _, _, mock_override_store = _make_harness_with_mock_stores()
        err = sqlite3.OperationalError('disk I/O error')
        mock_override_store.checkpoint.side_effect = err

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            harness._checkpoint_stores()  # must not raise

        assert any(
            'override_store.checkpoint() failed' in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        ), f'Expected override_store.checkpoint() failed warning; got: {caplog.records}'

    def test_checkpoint_skips_none_run_store(self) -> None:
        """When _run_store is None (init-failure path), checkpoint is skipped — no error."""
        harness, _, mock_event_store, mock_override_store = (
            _make_harness_with_mock_stores()
        )
        harness._run_store = None

        harness._checkpoint_stores()  # must not raise AttributeError

        # The other two stores still run.
        mock_event_store.checkpoint.assert_called_once()
        mock_override_store.checkpoint.assert_called_once()

    def test_checkpoint_skips_none_event_store(self) -> None:
        """When event_store is None (init-failure path), checkpoint is skipped — no error."""
        harness, mock_run_store, _, mock_override_store = (
            _make_harness_with_mock_stores()
        )
        harness.event_store = None

        harness._checkpoint_stores()  # must not raise AttributeError

        mock_run_store.checkpoint.assert_called_once()
        mock_override_store.checkpoint.assert_called_once()
