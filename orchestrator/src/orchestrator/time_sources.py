"""The default clocks behind an injectable time source, with one None-fallback rule per clock kind."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime


def resolve_monotonic_time_source(ts: Callable[[], float] | None) -> Callable[[], float]:
    """Return *ts* if provided, else the stdlib ``time.monotonic`` callable."""
    return ts if ts is not None else (lambda: time.monotonic())


def resolve_wall_time_source(
    ts: Callable[[], datetime] | None,
) -> Callable[[], datetime]:
    """Return *ts* if provided, else a real wall-clock ``datetime.now(UTC)``.

    Distinct from :func:`resolve_monotonic_time_source` (``time.monotonic``,
    which resets across process restarts): persisted wall-clock timestamps
    that must survive a restart, such as milestone anchors (task 2335 β),
    need a restart-safe, human-meaningful clock, so their callers inject or
    resolve one here instead.
    """
    return ts if ts is not None else (lambda: datetime.now(UTC))  # clock-exempt: single-capture writer
