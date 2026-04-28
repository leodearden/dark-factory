"""Tests for the _install_operator_stop_handler helper and main() exit-code logic
in fused_memory.server.main.

Fix B: differentiate operator-stop (clean exit 0, systemd does not restart)
from cascade-shutdown (exit 1, Restart=on-failure fires) via a module-level
flag set by the SIGTERM/SIGINT handler.
"""

import asyncio
import os
import signal
import sys
from unittest.mock import MagicMock, patch

import pytest

import fused_memory.server.main as main_mod
from fused_memory.server.main import _install_operator_stop_handler


@pytest.fixture(autouse=True)
def _reset_operator_flag():
    """Each test starts with _operator_stop_received=False and the SIGTERM/SIGINT
    handlers restored on exit so we don't pollute later tests."""
    main_mod._operator_stop_received = False
    prior_term = signal.getsignal(signal.SIGTERM)
    prior_int = signal.getsignal(signal.SIGINT)
    try:
        yield
    finally:
        main_mod._operator_stop_received = False
        signal.signal(signal.SIGTERM, prior_term)
        signal.signal(signal.SIGINT, prior_int)


class TestInstallOperatorStopHandlerHappyPath:
    def test_install_uses_add_signal_handler_for_sigterm_and_sigint(self):
        """Helper calls loop.add_signal_handler for both SIGTERM and SIGINT."""
        cb = MagicMock()
        mock_loop = MagicMock()

        with patch('asyncio.get_running_loop', return_value=mock_loop):
            _install_operator_stop_handler(cb)

        installed_signals = {
            call.args[0] for call in mock_loop.add_signal_handler.call_args_list
        }
        assert installed_signals == {signal.SIGTERM, signal.SIGINT}

    def test_handler_invocation_sets_flag_and_invokes_callback(self):
        """The captured handler sets _operator_stop_received and invokes the callback."""
        cb = MagicMock()
        mock_loop = MagicMock()

        with patch('asyncio.get_running_loop', return_value=mock_loop):
            _install_operator_stop_handler(cb)

        # Pull the captured handler — it's the same callable for SIGTERM/SIGINT.
        handler = mock_loop.add_signal_handler.call_args_list[0].args[1]
        handler('SIGTERM')

        assert main_mod._operator_stop_received is True
        cb.assert_called_once()


class TestInstallOperatorStopHandlerNoLoop:
    def test_no_running_loop_logs_warning_and_installs_nothing(self):
        cb = MagicMock()
        with patch(
            'asyncio.get_running_loop',
            side_effect=RuntimeError('no running event loop'),
        ), patch(
            'fused_memory.server.main.signal.signal'
        ) as mock_signal, patch(
            'fused_memory.server.main.logger'
        ) as mock_logger:
            _install_operator_stop_handler(cb)

        mock_signal.assert_not_called()
        cb.assert_not_called()
        assert main_mod._operator_stop_received is False
        mock_logger.warning.assert_called_once()


class TestInstallOperatorStopHandlerFallback:
    def test_falls_back_to_signal_signal_on_not_implemented(self):
        """add_signal_handler NotImplementedError (Windows) → use signal.signal."""
        cb = MagicMock()
        mock_loop = MagicMock()
        mock_loop.add_signal_handler.side_effect = NotImplementedError

        with patch('asyncio.get_running_loop', return_value=mock_loop), \
                patch('fused_memory.server.main.signal.signal') as mock_signal:
            _install_operator_stop_handler(cb)

        # Both signals must be installed via signal.signal as fallback.
        installed = {call.args[0] for call in mock_signal.call_args_list}
        assert installed == {signal.SIGTERM, signal.SIGINT}


@pytest.mark.skipif(
    sys.platform == 'win32',
    reason='os.kill SIGTERM/SIGINT semantics differ on Windows',
)
class TestInstallOperatorStopHandlerIntegration:
    """End-to-end: a real SIGTERM/SIGINT delivered via os.kill flips the flag."""

    def test_real_sigterm_sets_operator_flag(self):
        cb_invoked = []

        async def _run() -> None:
            received = asyncio.Event()

            def _cb() -> None:
                cb_invoked.append(True)
                received.set()

            _install_operator_stop_handler(_cb)
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.wait_for(received.wait(), timeout=2.0)

        with asyncio.Runner() as runner:
            runner.run(_run())

        assert main_mod._operator_stop_received is True
        assert cb_invoked == [True]

    def test_real_sigint_sets_operator_flag(self):
        cb_invoked = []

        async def _run() -> None:
            received = asyncio.Event()

            def _cb() -> None:
                cb_invoked.append(True)
                received.set()

            _install_operator_stop_handler(_cb)
            os.kill(os.getpid(), signal.SIGINT)
            await asyncio.wait_for(received.wait(), timeout=2.0)

        with asyncio.Runner() as runner:
            runner.run(_run())

        assert main_mod._operator_stop_received is True
        assert cb_invoked == [True]


class TestMainExitCode:
    """main() must exit 0 when the operator-stop flag is set, exit 1 otherwise.

    Tests use ``patch.object(os, '_exit')`` to capture the exit code without
    actually killing the test process.
    """

    def test_main_exits_0_when_operator_stop_received(self):
        async def _fake_run_server():
            main_mod._operator_stop_received = True

        with patch.object(main_mod, 'run_server', _fake_run_server), \
                patch.object(main_mod, '_acquire_singleton_lock'), \
                patch.object(main_mod, '_cancel_force_exit'), \
                patch.object(main_mod.os, '_exit') as mock_exit:
            main_mod.main()

        mock_exit.assert_called_once_with(0)

    def test_main_exits_1_on_cascade_cancellederror(self):
        async def _fake_run_server():
            raise asyncio.CancelledError

        with patch.object(main_mod, 'run_server', _fake_run_server), \
                patch.object(main_mod, '_acquire_singleton_lock'), \
                patch.object(main_mod, '_cancel_force_exit'), \
                patch.object(main_mod.os, '_exit') as mock_exit:
            main_mod.main()

        mock_exit.assert_called_once_with(1)

    def test_main_exits_1_on_unexpected_exception(self):
        async def _fake_run_server():
            raise RuntimeError('boom')

        with patch.object(main_mod, 'run_server', _fake_run_server), \
                patch.object(main_mod, '_acquire_singleton_lock'), \
                patch.object(main_mod, '_cancel_force_exit'), \
                patch.object(main_mod.os, '_exit') as mock_exit:
            main_mod.main()

        mock_exit.assert_called_once_with(1)

    def test_main_exits_0_on_keyboardinterrupt_pre_handler(self):
        """KeyboardInterrupt before the asyncio signal handler installs counts
        as an operator stop (Ctrl-C in the foreground is operator intent)."""
        async def _fake_run_server():
            raise KeyboardInterrupt

        with patch.object(main_mod, 'run_server', _fake_run_server), \
                patch.object(main_mod, '_acquire_singleton_lock'), \
                patch.object(main_mod, '_cancel_force_exit'), \
                patch.object(main_mod.os, '_exit') as mock_exit:
            main_mod.main()

        mock_exit.assert_called_once_with(0)

    def test_main_exits_1_when_run_server_returns_with_no_signal(self):
        """If run_server returns normally without the operator-stop handler firing
        (shouldn't happen in practice, but defends against a future regression),
        exit 1 — leave systemd to handle restart."""
        async def _fake_run_server():
            return None

        with patch.object(main_mod, 'run_server', _fake_run_server), \
                patch.object(main_mod, '_acquire_singleton_lock'), \
                patch.object(main_mod, '_cancel_force_exit'), \
                patch.object(main_mod.os, '_exit') as mock_exit:
            main_mod.main()

        mock_exit.assert_called_once_with(1)
