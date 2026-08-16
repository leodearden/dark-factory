"""Reconciliation drain/inflow throughput analysis (task 3049).

Answers three operator questions about a project's reconciliation pipeline:

* **inflow** — how many events actually arrive, per hour and per day, broken
  down by event type;
* **drain** — how much wall-clock the pipeline spends in each run mode
  (backlog chunk, steady state, remediation, targeted) and what that costs
  per event;
* **capacity** — the sustainable events/day that falls out of those two, and
  whether it clears the observed burst inflow.

Everything here is read-only over the reconciliation SQLite DB plus pure
arithmetic, so the report can be run out-of-process against a live production
database (see ``main`` / ``python -m``).

METHOD NOTE — the ISO8601 offset trap
-------------------------------------
``event_buffer.timestamp`` is an ISO8601 string that **carries an offset**::

    2026-07-25T13:23:22.383113+00:00

SQLite's ``datetime('now', ...)`` renders **space-separated and offset-free**::

    2026-07-25 13:23:22

Comparing the two in SQL (``WHERE timestamp < datetime('now', '-1 hour')``) is
a *string* comparison, and ``'T'`` (0x54) sorts above ``' '`` (0x20).  Every
same-day row therefore compares greater than the literal, so a whole day
collapses into a single bucket and the measurement silently reads as one giant
hour.  This is not hypothetical — it is the trap the task's METHOD NOTE calls
out.

The rule this module enforces: **bucket by parsing, never by SQL string
comparison against a ``datetime('now')`` literal.**  ``utc_hour_bucket`` is the
single place that parsing happens; every reader in this module and the rollup
in ``event_buffer.cleanup_drained`` route through it.
"""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = [
    'utc_hour_bucket',
]


def utc_hour_bucket(ts: str) -> str:
    """Normalise an ISO8601 timestamp to a UTC hour key ``'YYYY-MM-DDTHH'``.

    Parses (never string-compares — see the module METHOD NOTE), treats a naive
    timestamp as UTC, converts any offset-bearing timestamp *to* UTC rather
    than truncating at its local hour, and formats an hour key that sorts
    lexicographically in true chronological order.

    Args:
        ts: An ISO8601 timestamp, with or without an offset.

    Returns:
        The UTC hour key, e.g. ``'2026-07-25T13'``.

    Raises:
        ValueError: If ``ts`` is not a parseable ISO8601 timestamp.  Bucketing
            garbage into a plausible-looking wrong hour would silently corrupt
            the measurement, so this fails loudly instead.
    """
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).strftime('%Y-%m-%dT%H')
