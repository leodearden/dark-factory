"""A stateful task-store double with the scheduler's metadata-mode semantics.

``Scheduler.update_task`` merges a metadata patch into the stored record by
default and replaces it wholesale under ``metadata_mode='replace'``, the only
way to delete a key. This double holds task records in one dict and applies
those two semantics, so a harness under test reads back what it wrote.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock


class TaskStoreDouble:
    """Task records in one dict; every write is logged in ``calls``."""

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[Any, ...]] = []
        self.refused_metadata_keys: set[str] = set()

    def add(
        self, task_id: str, status: str = 'blocked', *,
        metadata: dict[str, Any] | None = None, **fields: Any,
    ) -> None:
        self.tasks[task_id] = {
            'id': task_id, 'status': status, 'metadata': dict(metadata or {}), **fields,
        }

    def wire_into(self, scheduler: Any) -> None:
        """Back *scheduler*'s task reads and writes with this store."""
        scheduler.get_task = AsyncMock(side_effect=self.get_task)
        scheduler.get_status = AsyncMock(side_effect=self.get_status)
        scheduler.update_task = AsyncMock(side_effect=self.update_task)
        scheduler.set_task_status = AsyncMock(side_effect=self.set_task_status)

    def metadata(self, task_id: str) -> dict[str, Any]:
        return self.tasks[task_id]['metadata']

    def status(self, task_id: str) -> str:
        return self.tasks[task_id]['status']

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        return copy.deepcopy(self.tasks.get(task_id))

    async def get_status(self, task_id: str) -> str | None:
        task = self.tasks.get(task_id)
        return None if task is None else task['status']

    async def update_task(
        self, task_id: str, metadata: dict[str, Any], *,
        append: bool = False, metadata_mode: str | None = None,
    ) -> bool:
        """Refuse (return False) a patch carrying any of ``refused_metadata_keys``."""
        self.calls.append(('update_task', task_id, copy.deepcopy(metadata), metadata_mode))
        if self.refused_metadata_keys & metadata.keys():
            return False
        task = self.tasks[task_id]
        if metadata_mode == 'replace':
            task['metadata'] = copy.deepcopy(metadata)
        else:
            task['metadata'] = {**task['metadata'], **copy.deepcopy(metadata)}
        return True

    async def set_task_status(self, task_id: str, status: str, **_: Any) -> None:
        self.calls.append(('set_task_status', task_id, status))
        self.tasks[task_id]['status'] = status

    def index_of(self, predicate: Callable[[tuple[Any, ...]], bool]) -> int:
        return next(i for i, call in enumerate(self.calls) if predicate(call))

    def writes_of(self, task_id: str, key: str) -> list[int]:
        """Indices in ``calls`` of the ``update_task`` patches that carried *key*."""
        return [
            i for i, call in enumerate(self.calls)
            if call[:2] == ('update_task', task_id) and key in call[2]
        ]
