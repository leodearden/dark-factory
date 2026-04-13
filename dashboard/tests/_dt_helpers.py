"""Shared datetime test helpers for merge-queue and related test modules."""

from __future__ import annotations

from datetime import UTC, datetime


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
