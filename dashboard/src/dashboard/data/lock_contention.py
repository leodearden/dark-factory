"""Who wants a module lock and who holds it: one rule for every dashboard tab that counts contention."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from shared.locking import modules_conflict


def lock_wanters(lock_sets: Iterable[Iterable[str]]) -> dict[str, int]:
    """How many members of a population want each module lock, each member counted once per key."""
    wanters: Counter[str] = Counter()
    for lock_set in lock_sets:
        wanters.update(set(lock_set))
    return dict(wanters)


def holder_of(key: str, current_holders: Mapping[str, str]) -> str | None:
    """The first holder, in *current_holders* order, whose lock conflicts with *key*.

    Conflict is the hierarchical prefix rule, so a holder of a sub-depth parent
    key holds its children.
    """
    return next(
        (holder for held, holder in current_holders.items() if modules_conflict(held, key)),
        None,
    )
