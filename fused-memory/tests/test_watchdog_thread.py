"""Tests for the dedicated-thread systemd WATCHDOG heartbeat helpers.

These tests guard the regression class: systemd SIGABRTs the whole
fused-memory.service when the asyncio event loop is starved/blocked >WatchdogSec
(task 1731 root-cause fix).  The heartbeat must run on a dedicated OS thread so
it fires independent of loop scheduling.

Step coverage:
  step-1  — _watchdog_thread_loop loop-independence test (RED → GREEN in step-2)
  step-3  — _start/_stop_watchdog_thread lifecycle tests (RED → GREEN in step-4)
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Step-1: loop-independence test for _watchdog_thread_loop
# ---------------------------------------------------------------------------

class TestWatchdogThreadLoop:
    """_watchdog_thread_loop must fire WATCHDOG=1 from a plain OS thread with
    NO asyncio event loop running.  This is the regression guard for the
    SIGABRT class: if the heartbeat shared the asyncio loop, a blocked loop
    would silence it and systemd would kill the process.
    """

    def test_pings_fire_without_asyncio_loop(self) -> None:
        """Pings must fire with no asyncio event loop running.

        Setup:
        - Monkeypatch main._sd_notify to a thread-safe counter list.
        - Spawn a daemon thread running _watchdog_thread_loop with a tiny
          interval (0.01 s) and a real threading.Event.
        - Poll for ≥2 calls (bounded 2 s) — then set the stop event and join.

        Assertions:
        - All recorded call args == 'WATCHDOG=1'     (correct message)
        - At least 2 pings fired                     (proves periodic behaviour)
        - Thread is not alive after stop             (clean shutdown)
        """
        import fused_memory.server.main as main  # noqa: PLC0415

        # Collect calls from the heartbeat thread into a thread-safe list.
        calls: list[str] = []
        calls_lock = threading.Lock()

        def mock_sd_notify(state: str) -> None:
            with calls_lock:
                calls.append(state)

        stop_event = threading.Event()

        with patch.object(main, '_sd_notify', mock_sd_notify):
            t = threading.Thread(
                target=main._watchdog_thread_loop,
                args=(stop_event, 0.01),
                daemon=True,
            )
            t.start()

            # Poll until ≥2 pings observed (max 2 s)
            deadline = time.monotonic() + 2.0
            while True:
                with calls_lock:
                    count = len(calls)
                if count >= 2:
                    break
                if time.monotonic() > deadline:
                    break
                time.sleep(0.005)

        # Signal stop and join before assertions (clean up)
        stop_event.set()
        t.join(timeout=1.0)

        # Assertions
        with calls_lock:
            snapshot = list(calls)

        assert len(snapshot) >= 2, (
            f"Expected ≥2 WATCHDOG=1 pings; got {len(snapshot)}.  "
            "Loop-independence regression: _watchdog_thread_loop must send pings "
            "from its OS thread independent of asyncio scheduling."
        )
        for arg in snapshot:
            assert arg == 'WATCHDOG=1', (
                f"All pings must send 'WATCHDOG=1'; got {arg!r}"
            )
        assert not t.is_alive(), (
            "Thread must not be alive after stop_event is set (clean shutdown)."
        )


# ---------------------------------------------------------------------------
# Step-3: lifecycle tests for _start_watchdog_thread / _stop_watchdog_thread
# (these tests are RED until step-4 adds the helpers)
# ---------------------------------------------------------------------------

class TestWatchdogThreadLifecycle:
    """Lifecycle contract for _start_watchdog_thread / _stop_watchdog_thread."""

    def _reset_module_refs(self, main: Any) -> None:
        """Reset module-level thread/event refs between test cases."""
        # Stop any running thread first
        if getattr(main, '_watchdog_thread', None) is not None:
            if getattr(main, '_watchdog_stop_event', None) is not None:
                main._watchdog_stop_event.set()
            t = main._watchdog_thread
            t.join(timeout=1.0)
        main._watchdog_thread = None
        main._watchdog_stop_event = None

    def test_start_with_notify_socket_spawns_daemon_thread(
        self, monkeypatch: Any
    ) -> None:
        """_start_watchdog_thread() with NOTIFY_SOCKET set returns True and
        spawns exactly one alive daemon thread."""
        import fused_memory.server.main as main  # noqa: PLC0415

        monkeypatch.setattr(main, '_sd_notify', lambda _: None)
        monkeypatch.setenv('NOTIFY_SOCKET', '/run/systemd/notify')
        self._reset_module_refs(main)

        result = main._start_watchdog_thread()

        assert result is True, "_start_watchdog_thread() must return True when started"
        assert main._watchdog_thread is not None, "_watchdog_thread ref must be set"
        assert main._watchdog_thread.is_alive(), "spawned thread must be alive"
        assert main._watchdog_thread.daemon is True, "thread must be a daemon thread"

        # Cleanup
        self._reset_module_refs(main)

    def test_start_idempotent_returns_false_on_second_call(
        self, monkeypatch: Any
    ) -> None:
        """A second _start_watchdog_thread() call must return False and NOT
        spawn a new thread while one is already running."""
        import fused_memory.server.main as main  # noqa: PLC0415

        monkeypatch.setattr(main, '_sd_notify', lambda _: None)
        monkeypatch.setenv('NOTIFY_SOCKET', '/run/systemd/notify')
        self._reset_module_refs(main)

        result1 = main._start_watchdog_thread()
        first_thread = main._watchdog_thread
        result2 = main._start_watchdog_thread()

        assert result1 is True
        assert result2 is False, "second call must return False (idempotent)"
        assert main._watchdog_thread is first_thread, (
            "module ref must still point to the original thread"
        )

        # Cleanup
        self._reset_module_refs(main)

    def test_start_without_notify_socket_is_no_op(
        self, monkeypatch: Any
    ) -> None:
        """_start_watchdog_thread() without NOTIFY_SOCKET returns False and
        does not spawn any thread."""
        import fused_memory.server.main as main  # noqa: PLC0415

        monkeypatch.setattr(main, '_sd_notify', lambda _: None)
        monkeypatch.delenv('NOTIFY_SOCKET', raising=False)
        self._reset_module_refs(main)

        result = main._start_watchdog_thread()

        assert result is False, (
            "_start_watchdog_thread() must return False when NOTIFY_SOCKET is unset"
        )
        assert main._watchdog_thread is None, (
            "_watchdog_thread must remain None when NOTIFY_SOCKET is unset"
        )

    def test_stop_clears_refs_and_thread_exits(
        self, monkeypatch: Any
    ) -> None:
        """_stop_watchdog_thread() sets the stop event, joins the thread, and
        clears both module refs so a subsequent _start_watchdog_thread() works."""
        import fused_memory.server.main as main  # noqa: PLC0415

        monkeypatch.setattr(main, '_sd_notify', lambda _: None)
        monkeypatch.setenv('NOTIFY_SOCKET', '/run/systemd/notify')
        self._reset_module_refs(main)

        main._start_watchdog_thread()
        thread_before = main._watchdog_thread
        assert thread_before is not None

        main._stop_watchdog_thread()

        assert not thread_before.is_alive(), (
            "thread must have exited after _stop_watchdog_thread()"
        )
        assert main._watchdog_thread is None, (
            "_watchdog_thread ref must be cleared to None after stop"
        )
        assert main._watchdog_stop_event is None, (
            "_watchdog_stop_event ref must be cleared to None after stop"
        )

    def test_stop_then_restart_works(
        self, monkeypatch: Any
    ) -> None:
        """After _stop_watchdog_thread(), a fresh _start_watchdog_thread()
        must succeed (restartable)."""
        import fused_memory.server.main as main  # noqa: PLC0415

        monkeypatch.setattr(main, '_sd_notify', lambda _: None)
        monkeypatch.setenv('NOTIFY_SOCKET', '/run/systemd/notify')
        self._reset_module_refs(main)

        r1 = main._start_watchdog_thread()
        main._stop_watchdog_thread()
        r2 = main._start_watchdog_thread()

        assert r1 is True
        assert r2 is True, (
            "_start_watchdog_thread() must return True after a prior stop (restartable)"
        )
        assert main._watchdog_thread is not None
        assert main._watchdog_thread.is_alive()

        # Cleanup
        self._reset_module_refs(main)
