"""Tests for shared.pytest_jobserver — timeout and lifecycle."""
from __future__ import annotations

import logging
import os
import threading

import pytest

import shared.pytest_jobserver as _js


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_globals():
    """Snapshot and restore plugin module-level globals around each test.

    The outer pytest session may have already acquired a jobserver token
    (because conftest.py lists pytest_jobserver as a pytest_plugins entry).
    Directly calling pytest_configure / pytest_unconfigure in these tests
    would clobber that state.  This fixture saves the current values, yields,
    and then restores them WITHOUT closing any outer-session fd.
    """
    saved_fd = _js._fd
    saved_tok = _js._tok
    yield
    _js._fd = saved_fd
    _js._tok = saved_tok


@pytest.fixture
def fifo_path(tmp_path):
    """Return path to a freshly-created FIFO under tmp_path."""
    path = tmp_path / 'test-jobserver'
    os.mkfifo(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Timeout-fallback tests
# ---------------------------------------------------------------------------


class TestTimeoutFallback:
    def test_timeout_fallback_when_fifo_has_no_writer(
        self, fifo_path, monkeypatch, caplog
    ):
        """pytest_configure must return (not hang) when no token is available.

        On the un-patched code this test fails because os.read blocks
        indefinitely; the fix uses select.select with a bounded timeout.
        The daemon=True thread ensures the blocked read dies with the process
        if this test regresses.
        """
        monkeypatch.setenv('PYTEST_JOBSERVER_FIFO', fifo_path)
        monkeypatch.setenv('PYTEST_JOBSERVER_TIMEOUT', '0.1')

        with caplog.at_level(logging.WARNING, logger='shared.pytest_jobserver'):
            thread = threading.Thread(
                target=_js.pytest_configure,
                kwargs={'config': None},
                daemon=True,
            )
            thread.start()
            thread.join(5.0)

        assert not thread.is_alive(), (
            'pytest_configure did not return within 5 s — '
            'unbounded os.read blocks indefinitely when no token is available'
        )
        assert _js._fd is None, 'Expected _fd to be reset to None after timeout'
        assert _js._tok is None, 'Expected _tok to be None after timeout'

        msgs = [r.getMessage() for r in caplog.records]
        assert any(fifo_path in m for m in msgs), (
            f'Expected FIFO path in warning message, got: {msgs}'
        )
        assert any('0.1' in m for m in msgs), (
            f'Expected timeout value in warning message, got: {msgs}'
        )


# ---------------------------------------------------------------------------
# _acquire_timeout_secs unit tests
# ---------------------------------------------------------------------------


class TestAcquireTimeoutSecs:
    """Unit-test env-var parsing without measuring wall time."""

    def test_default_when_key_absent(self):
        assert _js._acquire_timeout_secs({}) == 60.0

    def test_valid_float_string(self):
        assert _js._acquire_timeout_secs({'PYTEST_JOBSERVER_TIMEOUT': '0.25'}) == 0.25

    def test_invalid_string_falls_back_to_default(self):
        assert _js._acquire_timeout_secs({'PYTEST_JOBSERVER_TIMEOUT': 'abc'}) == 60.0

    def test_empty_string_falls_back_to_default(self):
        assert _js._acquire_timeout_secs({'PYTEST_JOBSERVER_TIMEOUT': ''}) == 60.0

    def test_integer_looking_string(self):
        assert _js._acquire_timeout_secs({'PYTEST_JOBSERVER_TIMEOUT': '5'}) == 5.0
