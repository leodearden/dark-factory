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
    """Snapshot, reset, and restore plugin module-level globals around each test.

    The outer pytest session may have already acquired a jobserver token
    (because conftest.py lists pytest_jobserver as a pytest_plugins entry).
    Directly calling pytest_configure / pytest_unconfigure in these tests
    would clobber that state.  This fixture:
      1. Saves the current values.
      2. Resets _fd/_tok to None so each test starts in a clean plugin state.
      3. Yields.
      4. Restores the saved values WITHOUT closing any outer-session fd.
    """
    saved_fd = _js._fd
    saved_tok = _js._tok
    _js._fd = None
    _js._tok = None
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
        assert any(f'{0.1:.1f}' in m for m in msgs), (
            f"Expected timeout value '{0.1:.1f}' in warning message, got: {msgs}"
        )


# ---------------------------------------------------------------------------
# _read_timeout_secs unit tests
# ---------------------------------------------------------------------------


class TestReadTimeoutSecs:
    """Unit-test env-var parsing without measuring wall time."""

    def test_default_when_key_absent(self):
        assert _js._read_timeout_secs({}) == 60.0

    def test_valid_float_string(self):
        assert _js._read_timeout_secs({'PYTEST_JOBSERVER_TIMEOUT': '0.25'}) == 0.25

    def test_invalid_string_falls_back_to_default(self):
        assert _js._read_timeout_secs({'PYTEST_JOBSERVER_TIMEOUT': 'abc'}) == 60.0

    def test_empty_string_falls_back_to_default(self):
        assert _js._read_timeout_secs({'PYTEST_JOBSERVER_TIMEOUT': ''}) == 60.0

    def test_integer_looking_string(self):
        assert _js._read_timeout_secs({'PYTEST_JOBSERVER_TIMEOUT': '5'}) == 5.0


# ---------------------------------------------------------------------------
# Happy-path regression test
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_acquires_token_when_fifo_is_seeded(
        self, fifo_path, monkeypatch, caplog
    ):
        """pytest_configure should acquire the token and register cleanup.

        atexit.register and signal.signal are monkeypatched to no-ops so the
        atexit callback and SIGTERM handler that pytest_configure installs on
        the happy path do not leak into the outer pytest process.  Without
        these patches the registrations would persist after the test because
        _restore_globals only restores _fd/_tok; they are harmless (both
        handlers are no-ops when _fd/_tok are None), but stubbing them out
        makes the test self-contained and avoids relying on that invariant.
        """
        monkeypatch.setenv('PYTEST_JOBSERVER_FIFO', fifo_path)
        # Use a generous timeout so the happy path never races
        monkeypatch.setenv('PYTEST_JOBSERVER_TIMEOUT', '5.0')

        # Suppress side-effecting registrations so they don't leak into the
        # outer pytest session's atexit stack and SIGTERM handler chain.
        monkeypatch.setattr(_js.atexit, 'register', lambda *a, **kw: None)
        monkeypatch.setattr(_js.signal, 'signal', lambda *a, **kw: None)

        # Seed one token into the FIFO before calling configure.
        # Open O_RDWR so the open itself doesn't block (same trick as the plugin),
        # then write a single byte.
        seed_fd = os.open(fifo_path, os.O_RDWR)
        os.write(seed_fd, b'x')

        try:
            with caplog.at_level(logging.WARNING, logger='shared.pytest_jobserver'):
                _js.pytest_configure(config=None)
        finally:
            os.close(seed_fd)

        assert _js._tok == b'x', f'Expected token b"x", got {_js._tok!r}'
        assert _js._fd is not None, 'Expected _fd to be set after successful acquire'

        # No timeout warning should have been emitted
        msgs = [r.getMessage() for r in caplog.records]
        assert not any('timed out' in m for m in msgs), (
            f'Unexpected timeout warning on happy path: {msgs}'
        )

        # Cleanup: unconfigure should return the token and reset state
        _js.pytest_unconfigure(config=None)
        assert _js._fd is None
        assert _js._tok is None


# ---------------------------------------------------------------------------
# Missing-FIFO regression test
# ---------------------------------------------------------------------------


class TestMissingFifo:
    def test_missing_fifo_is_silent_noop(
        self, tmp_path, monkeypatch, caplog
    ):
        """When the FIFO path does not exist, configure must be a silent no-op.

        The missing-FIFO early-return (pytest_jobserver.py:43-44) must NOT
        emit any warning — only the real timeout path logs.
        """
        nonexistent = str(tmp_path / 'no-such-fifo')
        monkeypatch.setenv('PYTEST_JOBSERVER_FIFO', nonexistent)

        with caplog.at_level(logging.WARNING, logger='shared.pytest_jobserver'):
            _js.pytest_configure(config=None)

        assert _js._fd is None
        assert _js._tok is None
        jobserver_warnings = [
            r for r in caplog.records
            if r.name == 'shared.pytest_jobserver' and r.levelno >= logging.WARNING
        ]
        assert jobserver_warnings == [], (
            f'Missing-FIFO path must not log warnings, got: {jobserver_warnings}'
        )


# ---------------------------------------------------------------------------
# Unconfigure-after-timeout safety test
# ---------------------------------------------------------------------------


class TestUnconfigureAfterTimeout:
    def test_pytest_unconfigure_after_timeout_is_safe(
        self, fifo_path, monkeypatch
    ):
        """pytest_unconfigure must be safe to call after the timeout path.

        The timeout path leaves _fd/_tok as None. Calling pytest_unconfigure
        afterwards must not raise and must leave _fd/_tok as None, guarding
        against half-initialised state bugs.
        """
        monkeypatch.setenv('PYTEST_JOBSERVER_FIFO', fifo_path)
        monkeypatch.setenv('PYTEST_JOBSERVER_TIMEOUT', '0.1')

        # Trigger the timeout path (run on daemon thread to avoid hanging)
        thread = threading.Thread(
            target=_js.pytest_configure,
            kwargs={'config': None},
            daemon=True,
        )
        thread.start()
        thread.join(5.0)
        assert not thread.is_alive(), 'pytest_configure hung unexpectedly'

        # State after timeout: _fd and _tok must both be None
        assert _js._fd is None
        assert _js._tok is None

        # Calling unconfigure on this state must not raise
        _js.pytest_unconfigure(config=None)

        assert _js._fd is None, '_fd should remain None after unconfigure'
        assert _js._tok is None, '_tok should remain None after unconfigure'
