"""D3's interval rules for every cumulative counter the recorder reads.

PRD ``plans/per-consumer-resource-recording-prd.md`` D3. A counter becomes an
increment over the interval since the last collection; an interval that is
missing, too old or not forward is a gap, and a gap writes no increment.
``sampler.consumers`` applies these rules to cgroup counters and
``sampler.host_memory`` to the reclaim threads' CPU.
"""

from __future__ import annotations

__all__ = ['counter_increment', 'is_gap']


def is_gap(previous_ts: int | None, now: int, nominal_interval: int) -> bool:
    """No previous collection, or one more than two intervals old (or not before *now*)."""
    return previous_ts is None or not 0 < now - previous_ts <= 2 * nominal_interval


def counter_increment(before: int | None, now: int) -> int:
    """current - previous, or the current value after a reset or when the counter is new."""
    return now if before is None or now < before else now - before
