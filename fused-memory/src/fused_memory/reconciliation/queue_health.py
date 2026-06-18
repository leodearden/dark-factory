"""Graphiti async-queue silent-drop diagnostics.

Surfaces the silent-failure tail of the MemoryService.add_memory() Graphiti path:
add_memory() returns success=True the instant a write is persisted to the
DurableWriteQueue SQLite row — BEFORE the worker actually executes the Graphiti
write.  If the worker fails repeatedly, the item is dead-lettered (status='dead')
after max_attempts, invisible to the original caller.

summarize_graphiti_queue_health() consumes the output of
DurableWriteQueue.get_stats(group_id=project_id) and produces a machine-readable
health record that reconciliation can surface in Stage 1 report.stats and WARNING
logs, making the silent-drop observable without any change to add_memory's public
interface.

Design: no I/O except for WARNING log on malformed input.  All other I/O is
performed by the harness in _check_graphiti_queue_health(); only interpretation
lives here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def summarize_graphiti_queue_health(stats: dict) -> dict:
    """Classify a DurableWriteQueue.get_stats() result into a health record.

    Args:
        stats: Dict from DurableWriteQueue.get_stats(group_id=...) with shape:
            {
                'counts': {status_str: int, ...},    # may be absent
                'oldest_pending_age_seconds': float | None,
            }
            Any missing keys default defensively so callers never see KeyError.

    Returns:
        dict with:
            dead_count (int): Number of dead-lettered writes (the silent-drop signal).
            pending_count (int): Writes waiting to be processed.
            retry_count (int): Writes being retried after a transient failure.
            oldest_pending_age_seconds (float | None): Age of the oldest pending/retry item.
            healthy (bool): True when dead_count == 0.

    Emits a WARNING log when *stats* is malformed (not a dict, or 'counts' key
    present but not a dict) and forces *healthy* to False in those cases.
    A benign-absent 'counts' key remains silent and healthy.
    """
    malformed = False

    if not isinstance(stats, dict):
        logger.warning(
            "summarize_graphiti_queue_health: stats is not a dict"
            " (type=%s, repr=%s); forcing healthy=False",
            type(stats).__name__,
            repr(stats)[:200],
        )
        malformed = True
        counts = {}
    else:
        raw_counts = stats.get('counts')
        if raw_counts is not None and not isinstance(raw_counts, dict):
            logger.warning(
                "summarize_graphiti_queue_health: stats['counts'] is present but"
                " not a dict (type=%s, repr=%s); forcing healthy=False",
                type(raw_counts).__name__,
                repr(raw_counts)[:200],
            )
            malformed = True
            counts = {}
        elif isinstance(raw_counts, dict):
            counts = raw_counts
        else:
            # 'counts' key absent — benign, stays silent
            counts = {}

    dead_count = int(counts.get('dead', 0))
    pending_count = int(counts.get('pending', 0))
    retry_count = int(counts.get('retry', 0))
    oldest_pending_age_seconds = (
        stats.get('oldest_pending_age_seconds') if isinstance(stats, dict) else None
    )

    healthy = (dead_count == 0) and not malformed

    return {
        'dead_count': dead_count,
        'pending_count': pending_count,
        'retry_count': retry_count,
        'oldest_pending_age_seconds': oldest_pending_age_seconds,
        'healthy': healthy,
    }
