"""Park refusals: a dispatch candidate refused its locks by parks alone (task 6328).

The class is a candidate whose modules are free of holders but covered by
active parks, none of which outranks it: a same-tier peer's park, or a
weaker one.  Only the scored scan's strict top earns skips and parks, so such
a candidate earns nothing while it waits, and waits as long as the park owner
does.  The scheduler makes that wait visible (``EventType.park_refused``)
rather than reserving for it.

Rationale: plans/scheduler-dispatch-scoring-and-lock-layer-prd.md §7,
"Annotation (task 6328)".  The operator lever: OPERATIONS.md §5, "A task
waiting on a peer's park".
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from orchestrator.pin_reservation import Blocker, BlockerKind, park_rank


def is_park_refusal(
    tier: str,
    blockers: Sequence[Blocker],
    top_park_rank: Callable[[str], int | None],
) -> bool:
    """True iff *blockers* are all parks, and no park owner outranks *tier*.

    *top_park_rank* answers the rank at which an owner's parks gate a
    requester, or None when the owner tops no park stack.
    """
    if not blockers:
        return False
    floor = park_rank(tier)
    for blocker in blockers:
        if blocker.kind is not BlockerKind.PARKED:
            return False
        rank = top_park_rank(blocker.owner)
        if rank is None or rank < floor:
            return False
    return True


@dataclass(frozen=True)
class _Episode:
    started_at: float
    last_emit_at: float | None = None


class ParkRefusalEpisodes:
    """The per-task cadence of ``park_refused``: one interval in, then per interval.

    An episode starts on a task's first observed park refusal.  It is due once
    *interval* seconds have passed since it started, and then once *interval*
    seconds have passed since its last emission.  Times are monotonic seconds.
    """

    def __init__(self) -> None:
        self._episodes: dict[str, _Episode] = {}

    def observe(self, task_id: str, *, now: float, interval: float) -> float | None:
        """The episode's age if *task_id* should emit now, else None.

        A non-None answer records the emission.
        """
        episode = self._episodes.get(task_id)
        if episode is None:
            self._episodes[task_id] = _Episode(started_at=now)
            return None
        since = episode.started_at if episode.last_emit_at is None else episode.last_emit_at
        if now - since < interval:
            return None
        self._episodes[task_id] = _Episode(episode.started_at, last_emit_at=now)
        return now - episode.started_at

    def end(self, task_id: str) -> None:
        """End *task_id*'s episode."""
        self._episodes.pop(task_id, None)

    def retain(self, task_ids: Iterable[str]) -> None:
        """End the episode of every tracked task not in *task_ids*."""
        keep = set(task_ids)
        for task_id in self._episodes.keys() - keep:
            del self._episodes[task_id]
