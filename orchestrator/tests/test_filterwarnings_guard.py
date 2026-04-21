"""Live guards for orchestrator/pyproject.toml filterwarnings entries.

These tests intentionally trigger the actual warning object via warnings.warn and
verify that the session-level filterwarnings in pyproject.toml promotes each to a
raised exception.

Pass condition  : filter regex is active → warning promoted to exception →
                  pytest.raises catches it → test green.
Failure condition: filter regex drifted (e.g. upstream reworded the message) →
                  warning is merely printed → pytest.raises gets no exception →
                  test red, surfacing the gap before real anti-patterns slip through CI.
"""

from __future__ import annotations

import warnings

import pytest


def test_asyncmock_coroutine_runtimewarning_filter_active():
    """RuntimeWarning filter for AsyncMock coroutine leaks must be live.

    The filter in pyproject.toml pins the exact CPython message text
    ``coroutine 'AsyncMockMixin._execute_mock_call' was never awaited``.
    If that text changes (e.g. mock module renamings, CPython rewording), the
    filter regex silently stops matching and real coroutine leaks stop failing CI.
    This test triggers the warning directly so pytest.raises can assert the filter
    is still active.
    """
    with pytest.raises(RuntimeWarning, match='was never awaited'):
        warnings.warn(
            "coroutine 'AsyncMockMixin._execute_mock_call' was never awaited",
            RuntimeWarning,
            stacklevel=1,
        )


def test_asyncio_mark_on_sync_function_filter_active():
    """PytestWarning filter for a sync def inside an @pytest.mark.asyncio class must be live.

    The filter pins the pytest-asyncio diagnostic text
    ``The test .* is marked with '@pytest.mark.asyncio' but it is not an async function``.
    If pytest-asyncio rewrites that message, the filter silently stops matching and the
    mark-mismatch anti-pattern stops failing CI.
    """
    with pytest.raises(pytest.PytestWarning, match="is marked with '@pytest.mark.asyncio'"):
        warnings.warn(
            "The test <Function foo> is marked with '@pytest.mark.asyncio' "
            'but it is not an async function. Please remove the asyncio mark.',
            pytest.PytestWarning,
            stacklevel=1,
        )
