"""Shared UTC timestamp utilities for the dashboard data layer."""

from __future__ import annotations

from datetime import UTC, datetime


def parse_utc(ts: str) -> datetime:
    """Parse an ISO timestamp string and ensure it has UTC timezone.

    Naive datetimes (no tzinfo) get UTC attached.  Aware datetimes are
    returned unchanged.  Invalid input propagates ValueError or TypeError
    from :func:`datetime.fromisoformat`.

    Args:
        ts: ISO 8601 timestamp string.

    Returns:
        A timezone-aware :class:`datetime` in UTC.
    """
    if ts is None:
        raise TypeError('timestamp is None')
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
