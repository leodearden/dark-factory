"""The task store's one timestamp form.

Every timestamp the task store writes carries exactly
``%Y-%m-%dT%H:%M:%S.mmmZ``: a UTC instant at millisecond precision with a
literal ``Z`` suffix, matching the Taskmaster ``updatedAt`` format. That holds
for ``updated_at``, ``heartbeat_at``, ``metadata.pending_since`` and
``created_at``, whichever writer produced the value: the live backend or an
offline back-fill.

The form is a CONTRACT, not an implementation detail. A writer that
substituted its own ``datetime.isoformat()`` would give one column two
incommensurable string shapes (offset-suffixed microseconds beside
``Z``-suffixed milliseconds), so lexical order would stop being instant order.
:func:`format_task_timestamp` is the single home of the form; callers supply
the instant.
"""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ['format_task_timestamp']


def format_task_timestamp(instant: datetime) -> str:
    """Render an aware *instant* in the task store's timestamp form.

    Microseconds are truncated, never rounded, to milliseconds, so a value
    never names a moment later than the instant it was read from. A naive
    datetime is refused: it names no instant, and guessing its zone would
    write a plausible but wrong time.
    """
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError(
            'task-store timestamps are UTC instants; a naive datetime has no instant: '
            f'{instant!r}'
        )
    utc = instant.astimezone(UTC)
    return utc.strftime('%Y-%m-%dT%H:%M:%S.') + f'{utc.microsecond // 1000:03d}Z'
