"""Shared UTC timestamp utilities for the dashboard data layer."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TypeVar, cast

_logger = logging.getLogger(__name__)
_T = TypeVar('_T')


def safe_gather_result(result: object, default: _T, label: str) -> _T:
    """Return *default* if *result* is an Exception, otherwise return *result*.

    Used with ``asyncio.gather(return_exceptions=True)`` to inspect each result
    independently, preserving sibling results when one coroutine fails.

    Non-Exception BaseExceptions (CancelledError, KeyboardInterrupt, SystemExit)
    are re-raised so that asyncio cancellation and process signals propagate
    correctly during shutdown and disconnect.

    Warnings are emitted under the ``dashboard.data.utils`` logger so that all
    gather-result warnings from any caller are routable via a single logger name.

    Args:
        result: The value returned by ``asyncio.gather(return_exceptions=True)``
            for one coroutine — may be a normal value or an exception instance.
        default: Value to return when *result* is an :class:`Exception`.
        label: Short string identifying the sub-query; included in the warning
            message to aid debugging.

    Returns:
        *result* cast to ``_T`` if it is not an exception, otherwise *default*.

    Raises:
        BaseException: Any non-:class:`Exception` BaseException (e.g.
            :exc:`asyncio.CancelledError`, :exc:`KeyboardInterrupt`,
            :exc:`SystemExit`) is re-raised immediately.
    """
    if isinstance(result, BaseException) and not isinstance(result, Exception):
        raise result
    if isinstance(result, Exception):
        _logger.warning('Error gathering %s: %s', label, result)
        return default
    return cast(_T, result)


def parse_utc(ts: str | None) -> datetime:
    """Parse an ISO timestamp string into a timezone-aware datetime.

    Naive datetimes (no tzinfo) get UTC attached.  Aware datetimes are
    returned unchanged.  None raises TypeError explicitly.  Other invalid
    input raises ValueError from :func:`datetime.fromisoformat`.

    Args:
        ts: ISO 8601 timestamp string.

    Returns:
        A timezone-aware :class:`datetime`.  Naive inputs are normalized to
        UTC; aware inputs are returned with their original tzinfo preserved
        (not necessarily UTC).  Callers that require UTC must call
        ``.astimezone(UTC)`` themselves.

    Note:
        The name ``parse_utc`` reflects only the naive-input default (naive
        datetimes get UTC attached).  It does **not** guarantee UTC output
        for aware inputs — those are returned with their original tzinfo.
    """
    if ts is None:
        raise TypeError('timestamp is None')
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
