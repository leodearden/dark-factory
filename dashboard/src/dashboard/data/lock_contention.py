"""Who wants a module lock, who holds it, and whether its holders fill its holder limit.

One rule for every dashboard tab that counts contention.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass

from shared.locking import modules_conflict


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

    def blocks(self, task_id: str) -> bool:
        """Whether the holders keep *task_id* off the key: they fill its limit and it is not one of them."""
        return self.at_limit and task_id not in self.holders


def key_holders(
    key: str,
    holders_by_key: Mapping[str, Collection[str]],
    holder_limit: Callable[[str], int],
) -> KeyHolders:
    """Who holds *key*, by the hierarchical prefix rule, and whether they fill its holder limit.

    A holder of a sub-depth parent key holds its children. The limit is
    *key*'s own when *key* is held, else the smallest among the held keys
    conflicting with it, since limits are published only for held keys.
    """
    conflicting = [held for held in holders_by_key if modules_conflict(held, key)]
    holders = tuple(sorted({task_id for held in conflicting for task_id in holders_by_key[held]}))
    if not holders:
        return KeyHolders(holders=(), at_limit=False)
    limit = holder_limit(key) if key in holders_by_key else min(map(holder_limit, conflicting))
    return KeyHolders(holders=holders, at_limit=len(holders) >= limit)
