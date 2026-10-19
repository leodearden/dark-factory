"""The deferral scripts' typed boundary over fused-memory's task tools.

Each reply is parsed here into a frozen value. Contract: plans/deferral-flip-condition-prd.md §9 ε.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from shared.task_deferral import Deferral, InvalidRecord, parse_record
from shared.task_metadata_wire import coerce_task_metadata
from shared.task_statuses import TaskStatus

__all__ = ['TaskRow']


def _task_id(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    raise ValueError(f'id {value!r} is not a task id')


def _title(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f'title {value!r} is not a string')
    return value


def _status(value: object) -> TaskStatus:
    try:
        return TaskStatus(value)
    except ValueError:
        raise ValueError(f'status {value!r} is not a task status') from None


@dataclass(frozen=True)
class TaskRow:
    """A task row as read over MCP; metadata None means its wire value was unreadable."""

    task_id: int
    title: str
    status: TaskStatus
    metadata: Mapping[str, Any] | None

    @classmethod
    def from_wire(cls, row: Mapping[str, Any]) -> TaskRow:
        metadata = coerce_task_metadata(row.get('metadata'))
        return cls(
            task_id=_task_id(row.get('id')),
            title=_title(row.get('title')),
            status=_status(row.get('status')),
            metadata=None if metadata is None else MappingProxyType(dict(metadata)),
        )

    @property
    def deferral(self) -> Deferral | InvalidRecord | None:
        if self.metadata is None:
            return InvalidRecord('metadata is not a JSON object')
        return parse_record(self.metadata.get('deferral'))
