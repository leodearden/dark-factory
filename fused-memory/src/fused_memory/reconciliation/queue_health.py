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


def _optional_counts(stats: dict, key: str) -> tuple[dict[str, int], bool]:
    """Read an optional ``{name: count}`` field of *stats* as ``(counts, malformed)``.

    Absent (or None) is benign: ``({}, False)``, silently.  Present but not a
    dict, or holding a count that is not int-convertible, is a malformed
    producer output: a WARNING and ``malformed`` True.  The returned dict is a
    fresh copy of the int-convertible entries only, so one bad count never
    costs the rest of the record.
    """
    raw = stats.get(key)
    if raw is None:
        return {}, False
    if not isinstance(raw, dict):
        logger.warning(
            "summarize_graphiti_queue_health: stats[%r] is present but not a dict"
            " (type=%s, repr=%s); forcing healthy=False",
            key,
            type(raw).__name__,
            repr(raw)[:200],
        )
        return {}, True

    counts: dict[str, int] = {}
    unconvertible: dict = {}
    for name, count in raw.items():
        try:
            counts[name] = int(count)
        except (TypeError, ValueError, OverflowError):
            unconvertible[name] = count
    if unconvertible:
        logger.warning(
            "summarize_graphiti_queue_health: stats[%r] holds non-integer counts"
            " (repr=%s); dropping them and forcing healthy=False",
            key,
            repr(unconvertible)[:200],
        )
    return counts, bool(unconvertible)


def summarize_graphiti_queue_health(stats: dict) -> dict:
    """Classify a DurableWriteQueue.get_stats() result into a health record.

    Args:
        stats: Dict from DurableWriteQueue.get_stats(group_id=...) with shape:
            {
                'counts': {status_str: int, ...},    # may be absent
                'oldest_pending_age_seconds': float | None,
                'dead_by_operation': {operation: int, ...},    # may be absent
            }
            Any missing keys default defensively so callers never see KeyError.

    Returns:
        dict with:
            dead_count (int): Number of dead-lettered writes (the silent-drop signal).
            pending_count (int): Writes waiting to be processed.
            retry_count (int): Writes being retried after a transient failure.
            oldest_pending_age_seconds (float | None): Age of the oldest pending/retry item.
            dead_by_operation (dict[str, int]): Dead writes per operation, {} when
                none or absent.
            healthy (bool): True when dead_count == 0.

    Emits a WARNING log when *stats* is malformed (not a dict, or a 'counts' or
    'dead_by_operation' key present but not a dict or holding a non-integer
    count) and forces *healthy* to False in those cases.  A non-integer count
    is dropped; the remaining counts still report.  A benign-absent 'counts' or
    'dead_by_operation' key remains silent and healthy.
    """
    stats_malformed = not isinstance(stats, dict)
    if stats_malformed:
        logger.warning(
            "summarize_graphiti_queue_health: stats is not a dict"
            " (type=%s, repr=%s); forcing healthy=False",
            type(stats).__name__,
            repr(stats)[:200],
        )
        stats = {}

    counts, counts_malformed = _optional_counts(stats, 'counts')
    dead_by_operation, breakdown_malformed = _optional_counts(stats, 'dead_by_operation')
    malformed = stats_malformed or counts_malformed or breakdown_malformed

    dead_count = counts.get('dead', 0)

    return {
        'dead_count': dead_count,
        'pending_count': counts.get('pending', 0),
        'retry_count': counts.get('retry', 0),
        'oldest_pending_age_seconds': stats.get('oldest_pending_age_seconds'),
        'dead_by_operation': dead_by_operation,
        'healthy': (dead_count == 0) and not malformed,
    }
