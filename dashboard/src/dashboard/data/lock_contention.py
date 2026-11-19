"""Who wants a module lock, who holds it, and whether its holders fill its holder limit.

One rule for every dashboard tab that counts contention.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from shared.locking import modules_conflict
from shared.scheduler_state import SchedulerStateSnapshot


def lock_wanters(lock_sets: Iterable[Iterable[str]]) -> dict[str, int]:
    """How many members of a population want each module lock, each member counted once per key."""
    wanters: Counter[str] = Counter()
    for lock_set in lock_sets:
        wanters.update(set(lock_set))
    return dict(wanters)


@dataclass(frozen=True, slots=True)
class KeyHolders:
    """Every task holding a lock that conflicts with one key, sorted, and whether they fill its limit."""

    holders: tuple[str, ...]
    at_limit: bool

    def __post_init__(self) -> None:
        if self.at_limit and not self.holders:
            raise ValueError(f'KeyHolders(holders={self.holders!r}, at_limit=True): no holders cannot fill a limit')
        if list(self.holders) != sorted(set(self.holders)):
            raise ValueError(f'KeyHolders.holders must be sorted and unique, got {self.holders!r}')

    def blocks(self, task_id: str) -> bool:
        """Whether the holders keep *task_id* off the key: they fill its limit and it is not one of them."""
        return self.at_limit and task_id not in self.holders


def key_holders_in(snapshot: SchedulerStateSnapshot) -> Callable[[str], KeyHolders]:
    """Who holds each key in *snapshot*, by the hierarchical prefix rule, and whether they fill its limit.

    A holder of a sub-depth parent key holds its children. A key's limit is
    its own when it is held, else the smallest among the held keys
    conflicting with it, since limits are published only for held keys.
    The snapshot's holders are read once, however many keys are asked about.
    """
    holders_by_key = snapshot.holders_by_key()

    def holders_of(key: str) -> KeyHolders:
        conflicting = [held for held in holders_by_key if modules_conflict(held, key)]
        holders = tuple(sorted({task_id for held in conflicting for task_id in holders_by_key[held]}))
        if not holders:
            return KeyHolders(holders=(), at_limit=False)
        limiting = [key] if key in holders_by_key else conflicting
        return KeyHolders(holders=holders, at_limit=len(holders) >= min(map(snapshot.holder_limit, limiting)))

    return holders_of
