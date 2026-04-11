"""Tests for fused_memory.utils.async_utils."""
from __future__ import annotations

import asyncio

import pytest

from fused_memory.utils.async_utils import propagate_cancellations


class TestPropagateCancellations:
    """Unit tests for propagate_cancellations().

    All tests are synchronous — propagate_cancellations is a sync guard helper
    that inspects already-settled gather() results.
    """

    def test_empty_sequence_returns_none(self) -> None:
        """Empty results list: no exception, returns None."""
        result = propagate_cancellations([])
        assert result is None

    def test_normal_values_return_none(self) -> None:
        """Sequence of plain values (int, str, None, dict): no raise, returns None."""
        result = propagate_cancellations([0, 'x', None, {'k': 1}])
        assert result is None

    def test_single_cancelled_error_raises(self) -> None:
        """Single CancelledError: must be re-raised."""
        exc = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            propagate_cancellations([exc])
        assert exc_info.value is exc

    def test_single_keyboard_interrupt_raises(self) -> None:
        """Single KeyboardInterrupt: must be re-raised."""
        exc = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt) as exc_info:
            propagate_cancellations([exc])
        assert exc_info.value is exc

    def test_single_system_exit_raises(self) -> None:
        """Single SystemExit: must be re-raised."""
        exc = SystemExit(0)
        with pytest.raises(SystemExit) as exc_info:
            propagate_cancellations([exc])
        assert exc_info.value is exc

    def test_single_runtime_error_returns_none(self) -> None:
        """RuntimeError is an Exception subclass — helper must NOT raise it."""
        result = propagate_cancellations([RuntimeError('boom')])
        assert result is None

    def test_runtime_error_then_cancelled_error_raises_cancelled(self) -> None:
        """[RuntimeError, CancelledError]: cancellation takes precedence over position.

        The helper scans the full sequence for bare-BaseException; even though
        RuntimeError appears first, CancelledError (not an Exception) must win.
        """
        runtime_err = RuntimeError('boom')
        cancelled = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            propagate_cancellations([runtime_err, cancelled])
        assert exc_info.value is cancelled

    def test_cancelled_error_then_runtime_error_raises_cancelled(self) -> None:
        """[CancelledError, RuntimeError]: cancellation-before-exception case."""
        cancelled = asyncio.CancelledError()
        runtime_err = RuntimeError('boom')
        with pytest.raises(asyncio.CancelledError) as exc_info:
            propagate_cancellations([cancelled, runtime_err])
        assert exc_info.value is cancelled

    def test_two_cancelled_errors_raises_first(self) -> None:
        """Two CancelledErrors: the FIRST one (by position) is raised — identity check."""
        first = asyncio.CancelledError()
        second = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            propagate_cancellations([first, second])
        assert exc_info.value is first

    def test_value_error_and_keyboard_interrupt_raises_keyboard_interrupt(self) -> None:
        """[ValueError, KeyboardInterrupt]: KeyboardInterrupt wins (bare BaseException)."""
        value_err = ValueError('bad value')
        ki = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt) as exc_info:
            propagate_cancellations([value_err, ki])
        assert exc_info.value is ki

    def test_tuple_input_accepted(self) -> None:
        """Helper accepts tuple input (Sequence, not just list)."""
        exc = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            propagate_cancellations((exc,))
        assert exc_info.value is exc

    def test_tuple_no_cancellation_returns_none(self) -> None:
        """Tuple with only normal values: returns None."""
        result = propagate_cancellations((0, 'x', RuntimeError('ok')))
        assert result is None

    def test_none_values_in_list_do_not_raise(self) -> None:
        """None is not a BaseException — must not trip the isinstance guard."""
        result = propagate_cancellations([None, None, None])
        assert result is None
