"""The single home of the durable write queue's row-status vocabulary.

Imported by fused-memory, whose ``DurableWriteQueue`` writes these statuses,
and by the dashboard, which reads them back from ``get_queue_stats``.
Not re-exported from ``shared/__init__.py``; import the full path.
"""

from __future__ import annotations

import enum

__all__ = ['WriteQueueStatus', 'CLAIMABLE', 'AWAITING_OPERATOR']


class WriteQueueStatus(enum.StrEnum):
    """Every status a ``write_queue`` row takes.

    Members are ``str`` instances, so one looks up the plain-str keyed
    ``counts`` dict ``get_stats`` emits.
    """

    PENDING = 'pending'
    RETRY = 'retry'
    PARKED = 'parked'
    IN_FLIGHT = 'in_flight'
    COMPLETED = 'completed'
    DEAD = 'dead'


#: The rows the queue still owes a write: a worker claims each once its
#: ``next_retry_at`` is due.
CLAIMABLE: tuple[WriteQueueStatus, ...] = (
    WriteQueueStatus.PENDING,
    WriteQueueStatus.RETRY,
    WriteQueueStatus.PARKED,
)

#: The rows only an operator moves on: a dead row was abandoned, and a parked
#: row is re-probed until a human clears its cause.
AWAITING_OPERATOR: frozenset[WriteQueueStatus] = frozenset(
    {WriteQueueStatus.PARKED, WriteQueueStatus.DEAD}
)
