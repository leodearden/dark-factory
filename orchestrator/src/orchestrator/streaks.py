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

import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)


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
        self.causes: dict[Any, str] = {}
        self.first_seen: dict[Any, float] = {}
        self.escalated: set[Any] = set()

    def bump(self, key: Any) -> int:
        """Increment the streak for ``key`` and return the new count."""
        value = self.counts.get(key, 0) + 1
        self.counts[key] = value
        return value

    def touch(self, key: Any, *, cause: str | None = None, now: float | None = None) -> Any:
        """Stamp/bump ``key`` — dispatches on whichever kwarg is given.

        ``now=`` selects the first-seen-age variant (mirrors the starvation
        watchdog): stamps ``first_seen[key]`` only the first time it is
        seen and is idempotent thereafter, returning the stored anchor.

        ``cause=`` selects the cause-change-reset variant (mirrors
        ``Scheduler._note_external_hold``): if the last-seen cause for
        ``key`` differs from ``cause`` (including the first-ever touch,
        where no cause has been seen), the streak resets to 0 before being
        bumped — so it always reflects a single consecutive-run cause
        rather than a mixed accumulation across differing causes.
        """
        if now is not None:
            if key not in self.first_seen:
                self.first_seen[key] = now
            return self.first_seen[key]
        if self.causes.get(key) != cause:
            self.counts[key] = 0
        self.causes[key] = cause
        value = self.counts.get(key, 0) + 1
        self.counts[key] = value
        return value

    def age(self, key: Any, now: float) -> float:
        """Elapsed time since ``key``'s first-seen anchor was stamped."""
        return now - self.first_seen.get(key, now)

    def mark_escalated(self, key: Any) -> None:
        """Record that ``key`` currently has an open escalation."""
        self.escalated.add(key)

    def is_escalated(self, key: Any) -> bool:
        """Whether ``key`` currently has an open escalation."""
        return key in self.escalated

    def value(self, key: Any) -> int:
        """Read the current streak for ``key`` (0 if never bumped)."""
        return self.counts.get(key, 0)

    def clear(self, key: Any) -> None:
        """Reset ``key`` — pop it from every backing container."""
        self.counts.pop(key, None)
        self.causes.pop(key, None)
        self.first_seen.pop(key, None)
        self.escalated.discard(key)

    def gc(self, stale_ids: Iterable[Any]) -> None:
        """Drop every key whose ``key_fn``-extracted task-id is stale.

        Mutates the backing containers IN PLACE — never rebinds — so
        aliased legacy attributes observe the sweep.
        """
        stale = stale_ids if isinstance(stale_ids, (set, frozenset)) else set(stale_ids)
        for key in [k for k in self.counts if self.key_fn(k) in stale]:
            del self.counts[key]
        for key in [k for k in self.causes if self.key_fn(k) in stale]:
            del self.causes[key]
        for key in [k for k in self.first_seen if self.key_fn(k) in stale]:
            del self.first_seen[key]
        for key in [k for k in self.escalated if self.key_fn(k) in stale]:
            self.escalated.discard(key)


class StreakRegistry:
    """Registry of StreakCounters swept together on every GC tick.

    Replaces the manual per-dict enumeration previously duplicated at each
    GC call site in ``Scheduler.acquire_next`` with one consolidated sweep,
    e.g. ``await registry.gc(_stale_ids, extra={'starvation': _non_eligible})``.
    The registry owns GC ONLY — counting and fire/escalate/resolve decisions
    stay with the counters and their call sites.
    """

    def __init__(self) -> None:
        self._counters: dict[str, tuple[StreakCounter, Callable[[Any], Awaitable[None]] | None]] = {}

    def register(
        self,
        name: str,
        counter: StreakCounter,
        *,
        on_gc: Callable[[Any], Awaitable[None]] | None = None,
    ) -> None:
        """Register ``counter`` under ``name`` for future ``gc()`` sweeps.

        ``on_gc``, if given, is awaited once per currently-escalated key
        being cleared by a sweep (see ``gc()``) — mirrors the starvation
        watchdog's resolve-then-discard GC behaviour.
        """
        self._counters[name] = (counter, on_gc)

    async def gc(
        self,
        stale_ids: Iterable[Any],
        *,
        extra: dict[str, Iterable[Any]] | None = None,
    ) -> None:
        """Sweep every registered counter, dropping stale (+ extra) ids.

        Each counter's sweep set is ``stale_ids``, unioned with
        ``extra[name]`` when the counter's registered name is a key in
        ``extra`` (e.g. the starvation counter's non-eligible-status ids).
        For a counter registered with ``on_gc``, the callback is awaited
        once for every key in ``counter.escalated`` whose ``key_fn``-
        extracted task-id falls in that counter's sweep set — BEFORE the
        counter itself is cleared, so the callback still observes the
        pre-GC state. A raising callback is logged and does not abort the
        sweep or skip clearing that key.
        """
        base_stale = stale_ids if isinstance(stale_ids, (set, frozenset)) else set(stale_ids)
        extra = extra or {}
        for name, (counter, on_gc) in self._counters.items():
            sweep_ids = base_stale | set(extra[name]) if name in extra else base_stale
            if not sweep_ids:
                continue
            if on_gc is not None:
                cleared_escalated = [
                    key for key in counter.escalated if counter.key_fn(key) in sweep_ids
                ]
                for key in cleared_escalated:
                    try:
                        await on_gc(key)
                    except Exception:
                        logger.warning(
                            'StreakRegistry on_gc callback for counter=%r key=%r '
                            'raised — continuing GC normally',
                            name,
                            key,
                            exc_info=True,
                        )
            counter.gc(sweep_ids)
