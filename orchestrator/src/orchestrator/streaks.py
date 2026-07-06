"""StreakCounter/StreakRegistry — uniform consecutive-tick counting + GC.

Consolidates the five hand-rolled "N consecutive ticks then fire, reset on
clean tick, GC on terminal" counters previously duplicated in scheduler.py:

- ``_external_unresolved_counts`` / ``_external_resolver_degraded_counts``
  (plain count variant)
- ``_external_hold_streak`` / ``_external_hold_cause`` (cause-change-reset
  variant)
- ``_local_backfill_unresolved_counts`` (plain count variant)
- ``_starvation_first_seen`` / ``_starvation_escalated`` (first-seen-age
  variant)

The registry owns COUNTING + GC ONLY.  Fire/escalate/block/resolve
decisions stay at the scheduler.py call sites, which read
``counter.value(key) >= threshold``.  See plans/supervision-quick-fixes-prd.md
task epsilon and task 2124.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


class StreakCounter:
    """A single consecutive-tick counter with pluggable variants.

    The base (count) variant is a plain "bump on each held tick, reset on
    each clean tick" counter keyed by an opaque key (str, or a
    ``(task_id, dep)`` tuple).  ``key_fn`` extracts the task-id component
    from a key for GC purposes (identity for str keys, ``lambda k: k[0]``
    for tuple keys) so ``gc()`` can drop exactly the entries belonging to a
    stale/terminal task.
    """

    def __init__(
        self,
        *,
        threshold: int | None = None,
        key_fn: Callable[[Any], Any] = lambda k: k,
    ) -> None:
        self.threshold = threshold
        self.key_fn = key_fn
        self.counts: dict[Any, int] = {}

    def bump(self, key: Any) -> int:
        """Increment the streak for ``key`` and return the new count."""
        value = self.counts.get(key, 0) + 1
        self.counts[key] = value
        return value

    def value(self, key: Any) -> int:
        """Read the current streak for ``key`` (0 if never bumped)."""
        return self.counts.get(key, 0)

    def clear(self, key: Any) -> None:
        """Reset ``key`` — pop it from the backing dict."""
        self.counts.pop(key, None)

    def gc(self, stale_ids: Iterable[Any]) -> None:
        """Drop every key whose ``key_fn``-extracted task-id is stale.

        Mutates the backing dict IN PLACE — never rebinds — so aliased
        legacy attributes observe the sweep.
        """
        stale = stale_ids if isinstance(stale_ids, (set, frozenset)) else set(stale_ids)
        for key in [k for k in self.counts if self.key_fn(k) in stale]:
            del self.counts[key]
