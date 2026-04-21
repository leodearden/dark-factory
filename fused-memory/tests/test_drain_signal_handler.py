"""Tests for the _register_drain_signal_handler helper in fused_memory.server.main."""

import signal
from unittest.mock import MagicMock, patch

import pytest

from fused_memory.server.main import _register_drain_signal_handler


class TestRegisterDrainSignalHandlerHappyPath:
    @pytest.mark.asyncio
    async def test_register_drain_signal_handler_uses_add_signal_handler(self):
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
    @pytest.mark.asyncio
    async def test_register_drain_signal_handler_falls_back_on_not_implemented(self):
        """Fallback: when add_signal_handler raises NotImplementedError, uses signal.signal."""
        reconciliation_harness = MagicMock()
        reconciliation_harness.drain = MagicMock()

        mock_loop = MagicMock()
        mock_loop.add_signal_handler.side_effect = NotImplementedError

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
