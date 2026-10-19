"""The deferral scripts' typed boundary over fused-memory's task tools.

Each reply is parsed here into a frozen value, and anything that is not a
usable answer raises StoreUnreachable, which never means "none" (INV-11).
Contract: plans/deferral-flip-condition-prd.md §9 ε.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, TypeVar

from dashboard.config import DEFAULT_FUSED_MEMORY_URLS
from legibility import census_trigger
from shared.project_layout import tasks_db
from shared.task_deferral import Deferral, InvalidRecord, parse_record
from shared.task_metadata_wire import coerce_task_metadata
from shared.task_statuses import TaskStatus

__all__ = [
    'DEFERRAL_KEY',
    'DeferralStore',
    'RecordHit',
    'StoreUnreachable',
    'TaskRow',
    'add_store_argument',
    'known_project',
    'store_from_args',
]

DEFERRAL_KEY: Final = 'deferral'
FIND_PAGE_SIZE: Final = 100
FIND_MAX_PAGES: Final = 200
FUSED_MEMORY_URL_ENV: Final = 'FUSED_MEMORY_MCP_URL'
_SHOWN_CHARS: Final = 200

_Parsed = TypeVar('_Parsed')


class StoreUnreachable(Exception):
    """A store call that answered nothing usable: the transport raised, the tool
    answered an error payload, or the reply was malformed."""


def _shown(value: object) -> str:
    text = repr(value)
    return text if len(text) <= _SHOWN_CHARS else f'{text[:_SHOWN_CHARS]}…'


def _task_id(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    raise ValueError(f'id {_shown(value)} is not a task id')


def _title(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f'title {_shown(value)} is not a string')
    return value


def _status(value: object) -> TaskStatus:
    try:
        return TaskStatus(value)
    except ValueError:
        raise ValueError(f'status {_shown(value)} is not a task status') from None


def _mapping(value: object, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{what} {_shown(value)} is not an object')
    return value


def _list(value: object, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f'{what} {_shown(value)} is not a list')
    return value


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
        return parse_record(self.metadata.get(DEFERRAL_KEY))


@dataclass(frozen=True)
class RecordHit:
    task_id: int
    status: TaskStatus
    title: str


@dataclass(frozen=True)
class _FindPage:
    hits: tuple[RecordHit, ...]
    has_more: bool


def _parse_deferred_rows(reply: Mapping[str, Any]) -> tuple[TaskRow, ...]:
    rows = tuple(TaskRow.from_wire(_mapping(row, 'task row')) for row in _list(reply.get('tasks'), 'tasks'))
    not_deferred = [row.task_id for row in rows if row.status is not TaskStatus.DEFERRED]
    if not_deferred:
        raise ValueError(f'a deferred-only read answered rows that are not deferred: {not_deferred}')
    return rows


def _parse_hit(row: Mapping[str, Any]) -> RecordHit:
    return RecordHit(
        task_id=_task_id(row.get('task_id')), status=_status(row.get('status')), title=_title(row.get('title'))
    )


def _parse_find_page(reply: Mapping[str, Any]) -> _FindPage:
    hits = tuple(_parse_hit(_mapping(row, 'result row')) for row in _list(reply.get('results'), 'results'))
    has_more = _mapping(reply.get('pagination'), 'pagination').get('has_more')
    if not isinstance(has_more, bool):
        raise ValueError(f'pagination.has_more {_shown(has_more)} is not a bool')
    if has_more and not hits:
        raise ValueError('pagination.has_more is true on a page that returned nothing')
    return _FindPage(hits, has_more)


def _parse_statuses(reply: Mapping[str, Any]) -> Mapping[int, TaskStatus]:
    statuses = _mapping(reply.get('statuses'), 'statuses')
    return MappingProxyType({_task_id(task_id): _status(status) for task_id, status in statuses.items()})


def _root_argument(root: Path) -> str:
    if not root.is_absolute():
        raise ValueError(f'project root {root} is not absolute')
    return str(root)


@dataclass(frozen=True)
class DeferralStore:
    """The deferral reads over a fused-memory MCP endpoint at *url*, posting through
    ``post(url, tool_name, arguments, *, timeout)``."""

    url: str
    post: Callable[..., Any]
    call_timeout_secs: float = 30.0

    def _call(self, tool: str, arguments: dict[str, Any], parse: Callable[[Mapping[str, Any]], _Parsed]) -> _Parsed:
        where = f'{tool} for {arguments["project_root"]}'
        try:
            reply = self.post(self.url, tool, arguments, timeout=self.call_timeout_secs)
        except Exception as exc:
            raise StoreUnreachable(f'{where}: the call raised {type(exc).__name__}: {exc}') from exc
        if isinstance(reply, Mapping) and 'error' in reply:
            raise StoreUnreachable(f'{where}: {reply.get("error_type", "error")}: {reply["error"]}')
        try:
            return parse(_mapping(reply, 'reply'))
        except ValueError as exc:
            raise StoreUnreachable(f'{where}: malformed reply: {exc}') from exc

    def deferred_rows(self, root: Path) -> tuple[TaskRow, ...]:
        arguments = {'project_root': _root_argument(root), 'statuses': [TaskStatus.DEFERRED.value]}
        return self._call('get_tasks', arguments, _parse_deferred_rows)

    def rows_with_record(self, root: Path) -> tuple[RecordHit, ...]:
        """Every row carrying a ``deferral`` key, in any status."""
        project_root = _root_argument(root)
        hits: list[RecordHit] = []
        for _ in range(FIND_MAX_PAGES):
            arguments = {'project_root': project_root, 'key': DEFERRAL_KEY, 'limit': FIND_PAGE_SIZE, 'offset': len(hits)}
            page = self._call('find_tasks_by_metadata', arguments, _parse_find_page)
            hits.extend(page.hits)
            if not page.has_more:
                return tuple(hits)
        raise StoreUnreachable(f'find_tasks_by_metadata for {project_root}: still has_more after {FIND_MAX_PAGES} pages')

    def statuses(self, root: Path, ids: Iterable[int]) -> Mapping[int, TaskStatus]:
        """Each found id's status; an id the store does not hold is absent."""
        project_root = _root_argument(root)
        wanted = sorted(set(ids))
        if not wanted:
            return MappingProxyType({})
        arguments = {'project_root': project_root, 'ids': [str(task_id) for task_id in wanted]}
        return self._call('get_statuses', arguments, _parse_statuses)


def known_project(root: Path) -> bool:
    """Whether *root* holds a task store; checked locally, before the server can answer an empty one."""
    return tasks_db(root).is_file()


def add_store_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--fused-memory-url',
        default=None,
        help=f'fused-memory base URL (default: ${FUSED_MEMORY_URL_ENV} or {DEFAULT_FUSED_MEMORY_URLS[0]})',
    )


def store_from_args(args: argparse.Namespace) -> DeferralStore:
    base = args.fused_memory_url or os.environ.get(FUSED_MEMORY_URL_ENV, DEFAULT_FUSED_MEMORY_URLS[0])
    return DeferralStore(url=f'{base.rstrip("/")}/mcp', post=census_trigger.post_mcp_tool_call)
