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

Design: pure, no I/O.  All I/O is performed by the harness in
_check_graphiti_queue_health(); only interpretation lives here.
"""

from __future__ import annotations


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

    Pure: no I/O, no side effects.
    """
    counts = stats.get('counts') if isinstance(stats, dict) else {}
    if not isinstance(counts, dict):
        counts = {}

    dead_count = int(counts.get('dead', 0))
    pending_count = int(counts.get('pending', 0))
    retry_count = int(counts.get('retry', 0))
    oldest_pending_age_seconds = (
        stats.get('oldest_pending_age_seconds') if isinstance(stats, dict) else None
    )

    return {
        'dead_count': dead_count,
        'pending_count': pending_count,
        'retry_count': retry_count,
        'oldest_pending_age_seconds': oldest_pending_age_seconds,
        'healthy': dead_count == 0,
    }
