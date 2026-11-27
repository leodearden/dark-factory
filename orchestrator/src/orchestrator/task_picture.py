"""The task picture: what the ordinary pass's read parts see at one moment.

plans/early-start-on-predecessor-branch-prd.md §5.4 "From what: the task
picture". ``Scheduler.build_task_picture`` builds one from the task fetch and
the read parts of the ordinary pass's writing steps, writing nothing; the
early-start coroutine keeps it for ``early_start.picture_max_age_secs``,
judging its age by ``built_at``.

It holds no locks, reservations, leases or waiting predecessors: those are
read at claim time, never from a picture. The maps that the dependency
predicate takes as ``dict`` stay ``dict``, and share nothing with the
scheduler (the dependency records are copies of its cached ones); treat them
as read-only. The other maps are read-only views.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from orchestrator.overrides import OverrideRow
from orchestrator.scheduler_priority_lookup import combine_priority, pin_orders
from orchestrator.task_priority import TaskPriority


@dataclass(frozen=True)
class TaskPicture:
    """The active tasks with their dependency facts, overrides in force and tiers.

    ``status_map`` includes the backfilled statuses of dependencies the active
    fetch leaves out; ``terminal_dep_records`` the full records of done or
    cancelled dependencies; ``delivered_check_cache`` is ``None`` while the
    delivered-check gate is switched off. ``built_at`` is on the scheduler's
    monotonic clock.
    """

    built_at: float
    tasks: tuple[dict, ...]
    status_map: dict[str, str]
    tasks_by_id: dict[str, dict]
    max_id: int
    terminal_dep_records: dict[str, dict]
    deferred_terminal_deps: frozenset[str]
    external_statuses: dict[str, str]
    external_resolver_failed: bool
    delivered_check_cache: dict[str, bool] | None
    overrides: Mapping[str, OverrideRow]
    effective_priorities: Mapping[str, str]
    transitive_counts: Mapping[str, int]

    def priority_of(self, task_id: str) -> TaskPriority:
        """*task_id*'s base tier in this picture combined with the pins in force."""
        return combine_priority(
            task_id, self.effective_priorities.get(task_id), pin_orders(self.overrides),
        )
