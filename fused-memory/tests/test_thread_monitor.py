"""Tests for thread monitor helpers in ``server/main.py``.

Covers:
  * ``_snapshot_threads()`` — categorises live threads by name prefix
  * ``_thread_monitor_iteration(prev, threshold) -> int`` — decides log level
    and calls _snapshot_threads when above threshold

Pattern mirrors tests/test_periodic_checkpoint.py (caplog-based logging tests).
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from fused_memory.server import main as server_main


class TestSnapshotThreads:
    """_snapshot_threads() returns a categorised count dict."""

    def test_returns_total_key_matching_enumerate(self):
        """_total must equal len(threading.enumerate())."""
        snapshot = server_main._snapshot_threads()
        assert '_total' in snapshot
        # Allow ±1 for thread churn between enumerate() calls
        assert abs(snapshot['_total'] - len(threading.enumerate())) <= 1

    def test_bucket_counts_sum_to_total(self):
        """Sum of all non-_total buckets must equal _total."""
        snapshot = server_main._snapshot_threads()
        total = snapshot['_total']
        bucket_sum = sum(v for k, v in snapshot.items() if k != '_total')
        assert bucket_sum == total

    def test_main_thread_bucketed_under_main(self):
        """MainThread must land in the 'main' bucket."""
        snapshot = server_main._snapshot_threads()
        assert snapshot.get('main', 0) >= 1

    def test_named_asyncio_thread_bucketed(self):
        """A thread named 'asyncio_test_X' must be counted in asyncio_pool."""
        barrier = threading.Barrier(2)
        stop_event = threading.Event()

        def worker():
            barrier.wait()
            stop_event.wait()

        t = threading.Thread(target=worker, name='asyncio_test_snapshot_X', daemon=True)
        t.start()
        barrier.wait()  # ensure the thread is alive before snapshotting
        try:
            snapshot = server_main._snapshot_threads()
            assert snapshot.get('asyncio_pool', 0) >= 1, (
                f"Expected asyncio_pool >= 1, got snapshot={snapshot}"
            )
        finally:
            stop_event.set()
            t.join(timeout=2)


class TestThreadMonitorIteration:
    """_thread_monitor_iteration(prev, threshold) -> int decides log level."""

    def test_no_change_below_threshold_no_log(self, monkeypatch, caplog):
        """count==prev and count<=threshold → no log records emitted."""
        monkeypatch.setattr(threading, 'active_count', lambda: 10)
        with caplog.at_level(logging.DEBUG, logger='fused_memory.server.main'):
            result = server_main._thread_monitor_iteration(prev=10, threshold=60)
        assert result == 10
        monitor_records = [r for r in caplog.records if 'thread_monitor' in r.getMessage()]
        assert len(monitor_records) == 0

    def test_change_below_threshold_emits_info(self, monkeypatch, caplog):
        """count!=prev and count<=threshold → exactly one INFO record."""
        monkeypatch.setattr(threading, 'active_count', lambda: 15)
        with caplog.at_level(logging.INFO, logger='fused_memory.server.main'):
            result = server_main._thread_monitor_iteration(prev=10, threshold=60)
        assert result == 15
        monitor_records = [r for r in caplog.records if 'thread_monitor' in r.getMessage()]
        assert len(monitor_records) == 1
        assert monitor_records[0].levelno == logging.INFO
        assert 'threads=15' in monitor_records[0].getMessage()
        assert 'delta=+5' in monitor_records[0].getMessage()

    def test_above_threshold_emits_warning_with_snapshot(self, monkeypatch, caplog):
        """count>threshold (delta=0) → WARNING with snapshot breakdown."""
        monkeypatch.setattr(threading, 'active_count', lambda: 70)
        with caplog.at_level(logging.WARNING, logger='fused_memory.server.main'):
            result = server_main._thread_monitor_iteration(prev=70, threshold=60)
        assert result == 70
        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and 'thread_monitor' in r.getMessage()
        ]
        assert len(warning_records) >= 1
        # The primary WARNING must contain the count
        assert any('threads=70' in r.getMessage() for r in warning_records)
        # A snapshot breakdown record must be emitted (contains _total=)
        all_records = [r for r in caplog.records if 'thread_monitor' in r.getMessage()]
        assert any('_total=' in r.getMessage() for r in all_records), (
            f"Expected snapshot breakdown in records: {[r.getMessage() for r in all_records]}"
        )

    def test_above_threshold_and_delta_uses_warning_not_info(self, monkeypatch, caplog):
        """count>threshold AND delta!=0 → WARNING (not INFO)."""
        monkeypatch.setattr(threading, 'active_count', lambda: 80)
        with caplog.at_level(logging.INFO, logger='fused_memory.server.main'):
            result = server_main._thread_monitor_iteration(prev=70, threshold=60)
        assert result == 80
        primary_records = [
            r for r in caplog.records
            if 'thread_monitor' in r.getMessage() and 'threads=80' in r.getMessage()
        ]
        assert len(primary_records) >= 1
        assert primary_records[0].levelno == logging.WARNING
