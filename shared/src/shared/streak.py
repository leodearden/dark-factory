"""Consecutive-failure streaks keyed by any hashable, firing once per run at a threshold.

Contract: plans/review-all-s14-shared-leaves-prd.md, decision 11 and invariant I7.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

K = TypeVar('K', bound=Hashable)


@dataclass(frozen=True)
class StreakStep:
    """One failure's outcome: the run's length so far, and whether it just reached the threshold."""

    count: int
    fired: bool


class KeyedStreaks(Generic[K]):
    """Per-key consecutive-failure counts; only keys in a live failure run hold state."""

    def __init__(self, *, threshold: int) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
            raise ValueError(
                'KeyedStreaks threshold must be an int >= 1 so a run of failures '
                f'can reach it; got threshold={threshold!r}'
            )
        self._threshold = threshold
        self._counts: dict[K, int] = {}

    def failure(self, key: K) -> StreakStep:
        """Extend *key*'s run by one; ``fired`` is true only at the call that reaches the threshold."""
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        return StreakStep(count=count, fired=count == self._threshold)

    def success(self, key: K) -> None:
        """End *key*'s run, re-arming it to fire again."""
        self._counts.pop(key, None)

    def count(self, key: K) -> int:
        """*key*'s current run length; 0 when it has none."""
        return self._counts.get(key, 0)

    def forget(self, keys: Iterable[K]) -> None:
        """Discard the runs of *keys*; unknown keys are ignored."""
        for key in keys:
            self._counts.pop(key, None)

    def retain(self, keep: Callable[[K], bool]) -> None:
        """Discard every run whose key *keep* rejects, deciding all keys before removing any."""
        doomed = [key for key in self._counts if not keep(key)]
        for key in doomed:
            del self._counts[key]
