"""Tests for the _register_drain_signal_handler helper in fused_memory.server.main."""

import asyncio
import os
import signal
import sys
from unittest.mock import MagicMock, patch

import pytest

from fused_memory.server.main import _register_drain_signal_handler


class TestRegisterDrainSignalHandlerHappyPath:
    def test_register_drain_signal_handler_uses_add_signal_handler(self):
        """Happy path: helper calls loop.add_signal_handler(SIGUSR1, cb) and cb triggers drain()."""
        reconciliation_harness = MagicMock()
        reconciliation_harness.drain = MagicMock()

        mock_loop = MagicMock()

        with patch('asyncio.get_running_loop', return_value=mock_loop):
            _register_drain_signal_handler(reconciliation_harness)

        mock_loop.add_signal_handler.assert_called_once()
        call_args = mock_loop.add_signal_handler.call_args
        assert call_args.args[0] == signal.SIGUSR1
        assert callable(call_args.args[1])

        # Invoke the captured callback and verify harness.drain() is called
        callback = call_args.args[1]
        callback()
        reconciliation_harness.drain.assert_called_once()


class TestRegisterDrainSignalHandlerFallback:
    @pytest.mark.parametrize('exc', [NotImplementedError, RuntimeError])
    def test_register_drain_signal_handler_falls_back_on_add_signal_handler_error(self, exc):
        """Fallback: when add_signal_handler raises NotImplementedError or RuntimeError, uses signal.signal.

        NotImplementedError covers Windows (no add_signal_handler support).
        RuntimeError covers non-main-thread usage (nested event loops, thread pools).
        """
        reconciliation_harness = MagicMock()
        reconciliation_harness.drain = MagicMock()

        mock_loop = MagicMock()
        mock_loop.add_signal_handler.side_effect = exc

        with patch('asyncio.get_running_loop', return_value=mock_loop), \
             patch('fused_memory.server.main.signal.signal') as mock_signal:
            _register_drain_signal_handler(reconciliation_harness)

        mock_signal.assert_called_once()
        call_args = mock_signal.call_args
        assert call_args.args[0] == signal.SIGUSR1
        assert callable(call_args.args[1])

        # Invoke the fallback callback with (signum, frame) shape required by signal.signal
        fallback_cb = call_args.args[1]
        fallback_cb(signal.SIGUSR1, None)
        reconciliation_harness.drain.assert_called_once()


class TestRegisterDrainSignalHandlerNoLoop:
    def test_register_drain_signal_handler_no_running_loop_installs_nothing(self):
        """When no event loop is running, no handler is installed and a warning is logged."""
        reconciliation_harness = MagicMock()

        with patch('asyncio.get_running_loop', side_effect=RuntimeError('no running event loop')), \
             patch('fused_memory.server.main.signal.signal') as mock_signal, \
             patch('fused_memory.server.main.logger') as mock_logger:
            _register_drain_signal_handler(reconciliation_harness)

        # No handler should be installed (early return)
        mock_signal.assert_not_called()
        reconciliation_harness.drain.assert_not_called()
        # A warning should have been logged
        mock_logger.warning.assert_called_once()


@pytest.mark.skipif(sys.platform == 'win32', reason='SIGUSR1 and loop.add_signal_handler are POSIX-only')
class TestRegisterDrainSignalHandlerIntegration:
    """End-to-end signal dispatch tests that send a real SIGUSR1 via os.kill.

    Each test uses asyncio.Runner for clean event-loop lifecycle management (no
    interaction with the thread's current-loop state or the deprecated event-loop
    policy API).  The SIGUSR1 handler is restored unconditionally in a try/finally.
    """

    def test_real_sigusr1_dispatch_triggers_drain(self):
        """Integration: a real SIGUSR1 delivered via os.kill reaches harness.drain() through asyncio."""
        stub_harness = MagicMock()
        stub_harness.drain = MagicMock()

        prior_handler = signal.getsignal(signal.SIGUSR1)
        try:
            async def _run() -> None:
                # Use an asyncio.Event so the test waits only as long as needed and
                # fails deterministically on timeout rather than a fixed sleep.
                drain_called = asyncio.Event()
                # Use call_soon_threadsafe for consistency with the fallback sibling test:
                # both integration tests share the same signal-safe wait pattern, making
                # the idiom grep-findable and future-proof for copy-paste into new signal tests.
                # (On the happy path drain.side_effect already runs in the loop thread via
                # asyncio's self-pipe, so this is cosmetic here — but keeps the pair uniform.)
                running_loop = asyncio.get_running_loop()
                stub_harness.drain.side_effect = lambda: running_loop.call_soon_threadsafe(drain_called.set)

                # Must run inside the running loop so asyncio.get_running_loop() resolves.
                _register_drain_signal_handler(stub_harness)
                # Verify the asyncio code path was taken (not the fallback):
                # loop.add_signal_handler internally installs asyncio.unix_events._sighandler_noop
                # via signal.signal; the fallback branch would install a plain lambda.
                # This check MUST be inside _run() before the Runner exits — loop.close()
                # calls remove_signal_handler() which restores SIG_DFL, destroying the evidence.
                current_handler = signal.getsignal(signal.SIGUSR1)
                assert current_handler is not prior_handler, "SIGUSR1 handler was not installed"
                assert getattr(current_handler, "__name__", None) != "<lambda>", \
                    "SIGUSR1 handler has fallback-lambda shape; expected asyncio-installed noop"
                os.kill(os.getpid(), signal.SIGUSR1)
                # Wait for asyncio's signal-dispatch machinery (self-pipe read +
                # call_soon_threadsafe callback) to invoke _handle_drain_signal.
                await asyncio.wait_for(drain_called.wait(), timeout=2.0)

            with asyncio.Runner() as runner:
                runner.run(_run())
            stub_harness.drain.assert_called_once()
        finally:
            signal.signal(signal.SIGUSR1, prior_handler)

    def test_real_sigusr1_fallback_dispatch_triggers_drain(self):
        """Integration: SIGUSR1 via the signal.signal fallback path reaches harness.drain().

        Forces the fallback branch by patching loop.add_signal_handler on the running loop
        to raise RuntimeError, then delivers a real signal and verifies end-to-end dispatch
        through the signal.signal code path (not the asyncio machinery).
        """
        stub_harness = MagicMock()
        stub_harness.drain = MagicMock()

        prior_handler = signal.getsignal(signal.SIGUSR1)
        try:
            async def _run() -> None:
                # Use an asyncio.Event for deterministic waiting (same as the happy-path test).
                drain_called = asyncio.Event()
                # Capture the running loop before setting side_effect: on the fallback path
                # the side_effect runs inside a real Python signal handler, so we must use
                # call_soon_threadsafe (asyncio.Event.set() internally uses loop.call_soon,
                # which is NOT signal-safe per CPython docs).
                running_loop = asyncio.get_running_loop()
                stub_harness.drain.side_effect = lambda: running_loop.call_soon_threadsafe(drain_called.set)

                # Force the fallback branch: make loop.add_signal_handler raise RuntimeError
                # so _register_drain_signal_handler falls back to signal.signal.
                # Spy on signal.signal with wraps= so the real installation still happens
                # (os.kill must actually dispatch to our handler below).
                with patch('fused_memory.server.main.signal.signal', wraps=signal.signal) as signal_signal_spy, \
                        patch.object(running_loop, 'add_signal_handler', side_effect=RuntimeError):
                    _register_drain_signal_handler(stub_harness)
                    signal_signal_spy.assert_called_once()
                    assert signal_signal_spy.call_args.args[0] == signal.SIGUSR1
                    assert callable(signal_signal_spy.call_args.args[1])

                os.kill(os.getpid(), signal.SIGUSR1)
                # The signal.signal handler is invoked synchronously at the next safe
                # bytecode boundary; wait_for handles both immediate and slightly deferred cases.
                await asyncio.wait_for(drain_called.wait(), timeout=2.0)

            with asyncio.Runner() as runner:
                runner.run(_run())
            stub_harness.drain.assert_called_once()
        finally:
            signal.signal(signal.SIGUSR1, prior_handler)
