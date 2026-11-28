"""What ``Scheduler.task_priority`` answers from between ticks.

The last-written tiers, read once at construction, and the lookup's own pin
map, kept apart from the tick's override snapshot.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

from shared.scheduler_state import MalformedSnapshotError, SchedulerStateSnapshot

from orchestrator.overrides import OverrideRow, OverrideStore
from orchestrator.task_priority import PINNED, TaskPriority

logger = logging.getLogger(__name__)

PAUSED_PIN_REFRESH_SECS = 60.0


def last_written_tiers(path: Path | None) -> dict[str, str]:
    """The base tiers in the scheduler_state.json at *path*; ``{}`` when there are none. Never raises.

    A ``PINNED`` entry names no base tier and is dropped: PRD D6 puts pins in
    ``effective_priorities``, and θ1 then owns restoring a pin's base tier.
    """
    if path is None:
        return {}
    try:
        snapshot = SchedulerStateSnapshot.from_json(json.loads(path.read_bytes()))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, MalformedSnapshotError) as exc:
        logger.warning('last_written_tiers: %s: %s; starting with no tiers', path, exc)
        return {}
    return {
        task_id: tier for task_id, tier in snapshot.effective_priorities.items() if tier != PINNED
    }


def pin_orders(rows: Mapping[str, OverrideRow]) -> dict[str, int]:
    """Every pinned row's pin order; a pinned row with no stored order counts as 0."""
    return {task_id: row.pin_order or 0 for task_id, row in rows.items() if row.pinned}


def combine_priority(
    task_id: str, base_tier: str | None, orders: Mapping[str, int] | None,
) -> TaskPriority:
    """*task_id*'s priority from its base tier and the pin orders; *orders* ``None`` means pins unknown.

    The one place a base tier is combined with pins.
    """
    if orders is None:
        return TaskPriority(tier=base_tier, pinned=None, base_tier=base_tier)
    order = orders.get(task_id)
    if order is None:
        return TaskPriority(tier=base_tier, pinned=False, base_tier=base_tier)
    return TaskPriority(tier=PINNED, pinned=True, base_tier=base_tier, pin_order=order)


class LookupPins:
    """The lookup's own pin map: task id to pin order, ``None`` while unknown.

    *current_store* is asked for the override store at every read, so the
    scheduler's own attribute stays its one owner.
    """

    def __init__(
        self,
        current_store: Callable[[], OverrideStore | None],
        project_root: str,
        *,
        clock: Callable[[], float],
    ) -> None:
        self._current_store = current_store
        self._project_root = project_root
        self._clock = clock
        self._orders: dict[str, int] | None = None
        self._read_at = clock()
        store = current_store()
        if store is not None:
            try:
                self._orders = pin_orders(store.get_overrides(project_root))
            except Exception:
                logger.warning(
                    'LookupPins: reading overrides for %s failed; pins unknown',
                    project_root, exc_info=True,
                )

    def adopt(self, rows: Mapping[str, OverrideRow]) -> None:
        if self._current_store() is not None:
            self._orders = pin_orders(rows)

    async def refresh_if_due(self) -> None:
        """Re-read the store off the event loop, at most once a :data:`PAUSED_PIN_REFRESH_SECS`. Never raises."""
        store = self._current_store()
        now = self._clock()
        if store is None or now - self._read_at < PAUSED_PIN_REFRESH_SECS:
            return
        self._read_at = now
        try:
            rows = await asyncio.to_thread(store.get_overrides, self._project_root)
        except Exception:
            logger.warning(
                'LookupPins: re-reading overrides for %s failed; keeping the last pins',
                self._project_root, exc_info=True,
            )
            return
        self._orders = pin_orders(rows)

    def priority(self, task_id: str, base_tier: str | None) -> TaskPriority:
        return combine_priority(task_id, base_tier, self._orders)
