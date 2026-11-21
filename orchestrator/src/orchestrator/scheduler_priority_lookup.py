"""The pin orders ``Scheduler.task_priority`` answers from, kept apart from the tick's override snapshot."""
from __future__ import annotations

from collections.abc import Callable, Mapping

from orchestrator.overrides import OverrideRow, OverrideStore
from orchestrator.task_priority import PINNED, TaskPriority


def pin_orders(rows: Mapping[str, OverrideRow]) -> dict[str, int]:
    """Every pinned row's pin order; a pinned row with no stored order counts as 0."""
    return {task_id: row.pin_order or 0 for task_id, row in rows.items() if row.pinned}


class LookupPins:
    """The lookup's own pin map: task id to pin order, ``None`` while unknown."""

    def __init__(
        self, store: OverrideStore | None, project_root: str, *, clock: Callable[[], float],
    ) -> None:
        self._store = store
        self._project_root = project_root
        self._clock = clock
        self._orders: dict[str, int] | None = None

    def adopt(self, rows: Mapping[str, OverrideRow]) -> None:
        if self._store is not None:
            self._orders = pin_orders(rows)

    def priority(self, task_id: str, base_tier: str | None) -> TaskPriority:
        if self._orders is None:
            return TaskPriority(tier=base_tier, pinned=None, base_tier=base_tier)
        order = self._orders.get(task_id)
        if order is None:
            return TaskPriority(tier=base_tier, pinned=False, base_tier=base_tier)
        return TaskPriority(tier=PINNED, pinned=True, base_tier=base_tier, pin_order=order)
