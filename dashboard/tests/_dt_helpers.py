"""The suite's clock helpers: fixed instants for the route clock and patched datetimes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from fastapi import FastAPI

from dashboard.api.clock import Clock, app_clock, install_clock


def fixed_clock(at: datetime) -> Clock:
    """Return a clock that always answers *at*."""
    return lambda: at


@contextmanager
def installed_clock(app: FastAPI, clock: Clock) -> Iterator[None]:
    """Serve *app*'s routes from *clock* for the block, then restore the previous clock."""
    previous = app_clock(app)
    install_clock(app, clock)
    try:
        yield
    finally:
        install_clock(app, previous)


def make_fixed_datetime_cls(fixed_now: datetime) -> type:
    """Return a ``datetime`` subclass whose ``.now(tz)`` always returns *fixed_now*.

    Use this in tests that need to mock ``datetime.now(UTC)`` without relying
    on a real wall clock.  The returned class asserts that *tz* is ``UTC``,
    catching accidental calls with a missing or unexpected timezone early.

    Usage::

        FIXED_NOW = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        _FixedDT = make_fixed_datetime_cls(FIXED_NOW)
        with patch.object(my_module, 'datetime', _FixedDT):
            result = fn_under_test()
        assert result == expected
    """

    class _FixedDT(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is UTC, f'Expected UTC, got {tz!r}'
            return fixed_now

    return _FixedDT
