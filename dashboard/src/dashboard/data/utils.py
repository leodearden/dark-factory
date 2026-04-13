"""Shared UTC timestamp utilities for the dashboard data layer."""

from __future__ import annotations

from datetime import UTC, datetime


def parse_utc(ts: str | None) -> datetime:
    """Parse an ISO timestamp string and ensure it has UTC timezone.

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
    """
    if ts is None:
        raise TypeError('timestamp is None')
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
