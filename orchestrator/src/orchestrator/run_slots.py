"""The harness's run slots: a general pool of ``max_concurrent_tasks`` and a reserve only a main-health fix may take.

A dispatch loop picks a task only once :meth:`RunSlots.free_scope` says a slot
can take it, so no task is held while the loop waits for a slot.
Rationale: ``docs/main-health-fix-run-slot-reserve.md``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from enum import Enum
from typing import Any

from orchestrator.main_health_fix_owner import is_main_health_fix


class DispatchScope(Enum):
    """Which tasks one dispatch tick may hand out, by the pool it dispatches into."""

    ANY = 'any'
    MAIN_HEALTH_FIX = 'main_health_fix'

    def admits(self, task: Mapping[str, Any]) -> bool:
        if self is DispatchScope.ANY:
            return True
        return is_main_health_fix(task)


_POOL_NAMES = {
    DispatchScope.ANY: 'general',
    DispatchScope.MAIN_HEALTH_FIX: 'main-health-fix reserve',
}


class RunSlotInvariantError(RuntimeError):
    """A claim that the loop's own ``free_scope`` check should have made impossible."""


class RunSlots:
    def __init__(self, general: int, main_health_fix_reserve: int) -> None:
        for argument, value in (('general', general), ('main_health_fix_reserve', main_health_fix_reserve)):
            if value < 0:
                raise ValueError(f'RunSlots {argument} must be >= 0, got {value}')
        self._pools = {
            DispatchScope.ANY: asyncio.Semaphore(general),
            DispatchScope.MAIN_HEALTH_FIX: asyncio.Semaphore(main_health_fix_reserve),
        }

    def free_scope(self) -> DispatchScope | None:
        """The widest scope with a free slot, or ``None`` when every slot is held."""
        for scope, pool in self._pools.items():
            if not pool.locked():
                return scope
        return None

    async def claim(self, scope: DispatchScope, task: Mapping[str, Any]) -> asyncio.Semaphore:
        """Take one slot of *scope*'s pool for *task*; the caller releases the returned semaphore."""
        task_id = task.get('id')
        if not scope.admits(task):
            raise RunSlotInvariantError(
                f'run slot claim refused: task {task_id} is not admitted by scope '
                f'{scope.value!r}',
            )
        pool = self._pools[scope]
        if pool.locked():
            raise RunSlotInvariantError(
                f'run slot claim refused: task {task_id} claimed scope {scope.value!r} '
                f'but the {_POOL_NAMES[scope]} pool is full',
            )
        await pool.acquire()
        return pool

