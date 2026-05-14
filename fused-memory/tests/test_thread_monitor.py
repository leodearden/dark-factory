"""Tests for thread monitor helpers in ``server/main.py``.

Covers:
  * ``_snapshot_threads()`` — categorises live threads by name prefix
  * ``_thread_monitor_iteration(prev, threshold) -> int`` — decides log level
    and calls _snapshot_threads when above threshold
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
