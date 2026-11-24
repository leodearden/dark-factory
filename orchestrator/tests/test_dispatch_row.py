"""refresh_dispatch_row: the claimed winner re-read from the store (task 11430)."""

from __future__ import annotations

import dataclasses

import pytest
from shared.task_fields import dependency_ids
from shared.task_read import TaskAbsent, TaskFound, TaskRead, TaskReadFailed

from orchestrator.dispatch_row import FreshRow, refresh_dispatch_row

pytestmark = pytest.mark.asyncio


class _Store:
    """A RowSource answering one read outcome and one statuses reply."""

    def __init__(
        self,
        read: TaskRead,
        statuses: dict[str, str] | None = None,
        statuses_error: Exception | None = None,
    ) -> None:
        self._read = read
        self._statuses = statuses or {}
        self._statuses_error = statuses_error
        self.read_ids: list[str] = []
        self.status_requests: list[list[str] | None] = []

    async def read_task(self, task_id: str, /) -> TaskRead:
        self.read_ids.append(task_id)
        return self._read

    async def get_statuses(
        self, ids: list[str] | None = None
    ) -> tuple[dict[str, str], Exception | None]:
        self.status_requests.append(ids)
        if self._statuses_error is not None:
            return {}, self._statuses_error
        return dict(self._statuses), None


def _row(**fields: object) -> dict:
    return {'id': '7', 'status': 'pending', 'dependencies': [], 'metadata': {}, **fields}


async def test_returns_the_store_row_and_its_dependency_statuses() -> None:
    snapshot = _row()
    stored = _row(dependencies=['9'], metadata={'gate_escalated_at': '2026-11-24T00:00:00Z'})
    store = _Store(TaskFound(stored), statuses={'9': 'pending'})

    fresh = await refresh_dispatch_row(store, snapshot)

    assert isinstance(fresh, FreshRow)
    assert fresh.task == stored
    assert fresh.task is not snapshot
    assert dict(fresh.dep_statuses) == {'9': 'pending'}
    assert store.status_requests == [dependency_ids(stored)]
    with pytest.raises(dataclasses.FrozenInstanceError):
        fresh.task = snapshot  # type: ignore[misc]


async def test_a_row_without_dependencies_reads_no_statuses() -> None:
    stored = _row()
    store = _Store(TaskFound(stored))

    fresh = await refresh_dispatch_row(store, _row())

    assert isinstance(fresh, FreshRow)
    assert dict(fresh.dep_statuses) == {}
    assert store.status_requests == []


async def test_an_absent_row_is_passed_through() -> None:
    absent = TaskAbsent()

    assert await refresh_dispatch_row(_Store(absent), _row()) is absent


async def test_a_failed_read_is_passed_through_without_reading_statuses() -> None:
    store = _Store(TaskReadFailed('fused-memory down'))

    outcome = await refresh_dispatch_row(store, _row())

    assert isinstance(outcome, TaskReadFailed)
    assert store.status_requests == []


async def test_unreadable_dependency_statuses_are_a_failed_read() -> None:
    store = _Store(
        TaskFound(_row(dependencies=['9'])), statuses_error=RuntimeError('timeout'),
    )

    outcome = await refresh_dispatch_row(store, _row())

    assert isinstance(outcome, TaskReadFailed)
    assert '7' in outcome.reason


async def test_an_int_id_is_read_as_its_string_form() -> None:
    store = _Store(TaskFound(_row()))

    await refresh_dispatch_row(store, _row(id=7))

    assert store.read_ids == ['7']
