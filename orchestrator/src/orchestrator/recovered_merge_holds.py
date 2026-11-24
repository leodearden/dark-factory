"""Restores at start-up the file holds of tasks whose merge request was recovered from the journal, and ends each when its request settles (task 10093)."""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol

from shared.task_statuses import TaskStatus

from orchestrator.module_charter import derive_modules

if TYPE_CHECKING:
    from orchestrator.event_store import EventStore
    from orchestrator.lock_table import UncontestedAcquisition

logger = logging.getLogger(__name__)

__all__ = [
    'HoldTasks',
    'LockRestoreReason',
    'RecoveredMergeHolds',
    'RestorableRequest',
]


class LockRestoreReason(enum.StrEnum):
    """The ``reason`` of a ``lock_acquired`` that restores a hold."""

    RECOVERED_MERGE = 'recovered_merge'


class _Outcome(Protocol):
    def done(self) -> bool: ...


class RestorableRequest(Protocol):
    @property
    def task_id(self) -> str: ...
    @property
    def request_id(self) -> str: ...
    @property
    def task_files(self) -> list[str] | None: ...
    @property
    def result(self) -> _Outcome: ...


class HoldTasks(Protocol):
    """What restoring and ending a hold needs of the task store; the Scheduler is one."""

    async def get_statuses(
        self, ids: list[str] | None = ...,
    ) -> tuple[dict[str, str], Exception | None]: ...
    def restore_hold(
        self, task_id: str, modules: list[str], /, *, reason: str,
    ) -> UncontestedAcquisition: ...
    def release(self, task_id: str, /) -> None: ...
    def is_dispatched(self, tid: str, /) -> bool: ...
    async def flush_state_snapshot(self) -> None: ...


class RecoveredMergeHolds:
    """Which task's locks are the restored hold of which recovered request."""

    def __init__(
        self, tasks: HoldTasks, *, lock_depth: int, event_store: EventStore | None,
    ) -> None:
        self._tasks = tasks
        self._lock_depth = lock_depth
        self._event_store = event_store
        self._restored: dict[str, str] = {}

    async def restore(self, requests: Iterable[RestorableRequest]) -> None:
        """Re-take each in-progress task's journalled files for its unfinished request."""
        wanted = [
            (request, keys) for request in requests
            if (keys := self._keys(request)) and not request.result.done()
        ]
        if not wanted:
            return
        statuses, _error = await self._tasks.get_statuses(
            [request.task_id for request, _keys in wanted],
        )
        restored = False
        for request, keys in wanted:
            restored |= self._restore_one(request, keys, statuses.get(request.task_id))
        if restored:
            await self._tasks.flush_state_snapshot()

    def _keys(self, request: RestorableRequest) -> list[str]:
        return derive_modules(
            request.task_files or [], self._lock_depth, task_id=request.task_id,
        )

    def _restore_one(
        self, request: RestorableRequest, keys: list[str], status: str | None,
    ) -> bool:
        tid = request.task_id
        if status != TaskStatus.IN_PROGRESS:
            logger.info(
                'Task %s is %s, not in progress: recovered merge request %s '
                'restores no hold', tid, status, request.request_id,
            )
            return False
        acquisition = self._tasks.restore_hold(
            tid, keys, reason=LockRestoreReason.RECOVERED_MERGE,
        )
        if not acquisition.acquired:
            return False
        self._restored[tid] = request.request_id
        logger.info(
            'Task %s re-took %s for recovered merge request %s',
            tid, list(acquisition.acquired), request.request_id,
        )
        return True
