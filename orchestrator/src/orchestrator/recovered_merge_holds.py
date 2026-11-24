"""Restores at start-up the file holds of tasks whose merge request was recovered from the journal, and ends each when its request settles (task 10093)."""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Protocol

from shared.task_statuses import TaskStatus

from orchestrator.event_store import EventType
from orchestrator.module_charter import derive_modules

if TYPE_CHECKING:
    from orchestrator.event_store import EventStore
    from orchestrator.lock_table import UncontestedAcquisition
    from orchestrator.pin_reservation import Blocker

logger = logging.getLogger(__name__)

__all__ = [
    'HoldShortfall',
    'HoldTasks',
    'LockRestoreReason',
    'RecoveredMergeHolds',
    'RestorableRequest',
]


class LockRestoreReason(enum.StrEnum):
    """The ``reason`` of a ``lock_acquired`` that restores a hold."""

    RECOVERED_MERGE = 'recovered_merge'


class HoldShortfall(enum.StrEnum):
    """The ``cause`` of a ``recovered_merge_hold_incomplete``: why the task holds less than its journalled keys."""

    CONTESTED = 'contested'
    STATUS_UNREAD = 'status_unread'
    RAISED = 'raised'


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
    def holds_restored(self, tid: str, /) -> bool: ...
    def release_restored_hold(self, task_id: str, /) -> bool: ...
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

    def holds(self, task_id: str, request_id: str) -> bool:
        """True while *task_id*'s locks are this request's restored hold, never again once it was dispatched."""
        return (
            self._restored.get(task_id) == request_id
            and self._tasks.holds_restored(task_id)
        )

    def release(self, task_id: str, request_id: str) -> None:
        """End this request's restored hold; a task dispatched since keeps whatever its workflows left."""
        if self._restored.get(task_id) != request_id:
            return
        del self._restored[task_id]
        self._tasks.release_restored_hold(task_id)

    async def restore(self, requests: Iterable[RestorableRequest]) -> None:
        """Re-take each in-progress task's journalled files for its unfinished request.

        Never raises: a failure restores less, so start-up goes on, and emits a
        ``recovered_merge_hold_incomplete`` naming its cause for each request it cost.
        """
        wanted = self._wanted(requests)
        if not wanted:
            return
        statuses, unread = await self._read_statuses([request.task_id for request, _keys in wanted])
        if unread is not None:
            for request, keys in wanted:
                self._report_shortfall(
                    request, HoldShortfall.STATUS_UNREAD, unheld=keys, error=unread,
                )
            return
        restored = False
        for request, keys in wanted:
            restored |= self._restore_contained(request, keys, statuses.get(request.task_id))
        if restored:
            await self._flush_snapshot()

    def _wanted(
        self, requests: Iterable[RestorableRequest],
    ) -> list[tuple[RestorableRequest, list[str]]]:
        wanted: list[tuple[RestorableRequest, list[str]]] = []
        for request in requests:
            try:
                keys = derive_modules(
                    request.task_files or [], self._lock_depth, task_id=request.task_id,
                )
                if keys and not request.result.done():
                    wanted.append((request, keys))
            except Exception as exc:
                logger.warning(
                    'Recovered merge request %s: deriving its task\'s lock keys '
                    'raised; no hold restored', request.request_id, exc_info=True,
                )
                self._report_shortfall(request, HoldShortfall.RAISED, unheld=(), error=repr(exc))
        return wanted

    async def _read_statuses(self, ids: list[str]) -> tuple[dict[str, str], str | None]:
        """The tasks' statuses, and the read's error when they could not be read."""
        try:
            statuses, error = await self._tasks.get_statuses(ids)
        except Exception as exc:
            logger.warning(
                'Reading the statuses of recovered merge requests\' tasks %s raised; '
                'no hold restored', ids, exc_info=True,
            )
            return {}, repr(exc)
        if error is not None:
            logger.warning(
                'Reading the statuses of recovered merge requests\' tasks %s failed '
                '(%r); no hold restored', ids, error,
            )
            return {}, repr(error)
        return statuses, None

    def _restore_contained(
        self, request: RestorableRequest, keys: list[str], status: str | None,
    ) -> bool:
        try:
            return self._restore_one(request, keys, status)
        except Exception as exc:
            logger.warning(
                'Task %s: restoring the hold of recovered merge request %s raised',
                request.task_id, request.request_id, exc_info=True,
            )
            self._report_shortfall(request, HoldShortfall.RAISED, unheld=keys, error=repr(exc))
            return False

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
        if request.result.done():
            logger.info(
                'Recovered merge request %s finalized while statuses were read: '
                'task %s restores no hold', request.request_id, tid,
            )
            return False
        acquisition = self._tasks.restore_hold(
            tid, keys, reason=LockRestoreReason.RECOVERED_MERGE,
        )
        if acquisition.acquired:
            self._restored[tid] = request.request_id
            logger.info(
                'Task %s re-took %s for recovered merge request %s',
                tid, list(acquisition.acquired), request.request_id,
            )
        if acquisition.contested:
            unheld = list(dict.fromkeys(blocker.module for blocker in acquisition.contested))
            logger.warning(
                'Task %s could not re-take %s for recovered merge request %s: '
                'another task holds or parks them', tid, unheld, request.request_id,
            )
            self._report_shortfall(
                request, HoldShortfall.CONTESTED, unheld=unheld,
                restored=acquisition.acquired, blockers=acquisition.contested,
            )
        return bool(acquisition.acquired)

    def _report_shortfall(
        self, request: RestorableRequest, cause: HoldShortfall, *,
        unheld: Sequence[str], restored: Sequence[str] = (),
        blockers: Sequence[Blocker] = (), error: str | None = None,
    ) -> None:
        if self._event_store is None:
            return
        self._event_store.emit(
            EventType.recovered_merge_hold_incomplete,
            task_id=request.task_id,
            data={
                'request_id': request.request_id,
                'cause': cause,
                'restored': sorted(restored),
                'unheld': sorted(unheld),
                'blockers': [blocker.as_payload() for blocker in blockers],
                'error': error,
            },
        )

    async def _flush_snapshot(self) -> None:
        try:
            await self._tasks.flush_state_snapshot()
        except Exception:
            logger.warning(
                'Flushing the scheduler snapshot after restoring recovered merge '
                'holds raised', exc_info=True,
            )
