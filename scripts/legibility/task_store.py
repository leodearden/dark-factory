"""The census's typed boundary over the fused-memory task tools it reads and
resolves through: ``find_tasks_by_metadata``, ``search_tasks``, ``get_task``
and ``resolve_ticket`` (fused-memory/src/fused_memory/server/tools.py). Each
reply is parsed here into a frozen value, and anything that is not a usable
answer raises StoreUnreachable, which never means "no match"
(plans/census-incremental-prd.md L4; docs/quality-findings-contract.md §8).
"""
from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, TypeVar

FIND_PAGE_SIZE = 100
FIND_MAX_PAGES = 200
RESOLVE_TRANSPORT_MARGIN_SECS = 15.0

_Parsed = TypeVar("_Parsed")


class StoreUnreachable(Exception):
    """A task-store call that answered nothing usable: the transport raised,
    the tool answered an error payload, or the reply was malformed. It never
    means "no match"."""


class _MalformedReply(ValueError):
    pass


class TicketStatus(enum.StrEnum):
    CREATED = "created"
    COMBINED = "combined"
    REFUSED = "refused"
    FAILED = "failed"


@dataclass(frozen=True)
class TaskHit:
    task_id: int
    title: str
    status: str


@dataclass(frozen=True)
class SearchHit:
    task_id: int
    title: str
    status: str | None
    score: float


@dataclass(frozen=True)
class TaskRecord:
    task_id: int
    title: str
    status: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TicketOutcome:
    status: TicketStatus
    task_id: int | None
    reason: str | None

    @property
    def undecided(self) -> bool:
        """The wait expired before the curator decided; the ticket may still be decided."""
        return self.status is TicketStatus.FAILED and self.reason == "timeout"


class TaskStore(Protocol):
    def find_tasks_by_metadata(self, project_root: str, *, key: str, value: str | int) -> tuple[TaskHit, ...]: ...

    def search_tasks(self, project_root: str, *, query: str, score_threshold: float) -> tuple[SearchHit, ...]: ...

    def get_task(self, project_root: str, task_id: int) -> TaskRecord: ...

    def resolve_ticket(self, project_root: str, ticket: str, *, timeout_secs: float) -> TicketOutcome: ...


def _task_id(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise _MalformedReply(f"task id {value!r} is not an integer")


def _text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise _MalformedReply(f"{field} {value!r} is not a string")
    return value


def _optional_text(row: Mapping[str, Any], field: str) -> str | None:
    value = row.get(field)
    return value if isinstance(value, str) else None


def _mapping(value: object, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _MalformedReply(f"{what} {value!r} is not an object")
    return value


def _rows(reply: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = reply.get("results")
    if not isinstance(rows, list):
        raise _MalformedReply(f"results {rows!r} is not a list")
    return [_mapping(row, "result row") for row in rows]


@dataclass(frozen=True)
class _FindPage:
    hits: tuple[TaskHit, ...]
    has_more: bool


def _parse_find_page(reply: Mapping[str, Any]) -> _FindPage:
    hits = tuple(
        TaskHit(task_id=_task_id(row.get("task_id")), title=_text(row, "title"), status=_text(row, "status"))
        for row in _rows(reply)
    )
    has_more = _mapping(reply.get("pagination"), "pagination").get("has_more")
    if not isinstance(has_more, bool):
        raise _MalformedReply(f"pagination.has_more {has_more!r} is not a bool")
    if has_more and not hits:
        raise _MalformedReply("pagination.has_more is true on a page that returned nothing")
    return _FindPage(hits, has_more)


def _score(row: Mapping[str, Any]) -> float:
    score = row.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise _MalformedReply(f"score {score!r} is not a number")
    return float(score)


def _parse_search(reply: Mapping[str, Any]) -> tuple[SearchHit, ...]:
    return tuple(
        SearchHit(
            task_id=_task_id(row.get("task_id")),
            title=_text(row, "title"),
            status=_optional_text(row, "status"),
            score=_score(row),
        )
        for row in _rows(reply)
    )


def _parse_task(reply: Mapping[str, Any]) -> TaskRecord:
    metadata = reply.get("metadata")
    return TaskRecord(
        task_id=_task_id(reply.get("id")),
        title=_text(reply, "title"),
        status=_text(reply, "status"),
        metadata=MappingProxyType(dict(_mapping({} if metadata is None else metadata, "metadata"))),
    )


def _parse_ticket(reply: Mapping[str, Any]) -> TicketOutcome:
    try:
        status = TicketStatus(reply.get("status"))
    except ValueError:
        raise _MalformedReply(f"status {reply.get('status')!r} is not a resolve_ticket status") from None
    decided_a_task = status in (TicketStatus.CREATED, TicketStatus.COMBINED)
    return TicketOutcome(
        status=status,
        task_id=_task_id(reply.get("task_id")) if decided_a_task else None,
        reason=_optional_text(reply, "reason"),
    )


@dataclass(frozen=True)
class McpTaskStore:
    """TaskStore over a fused-memory MCP endpoint at *url*, posting through
    ``post(url, tool_name, arguments, *, timeout)``."""

    url: str
    post: Callable[..., Any]
    call_timeout_secs: float = 30.0

    def _call(
        self,
        tool: str,
        arguments: dict[str, Any],
        parse: Callable[[Mapping[str, Any]], _Parsed],
        *,
        timeout: float | None = None,
    ) -> _Parsed:
        try:
            reply = self.post(
                self.url, tool, arguments, timeout=self.call_timeout_secs if timeout is None else timeout,
            )
        except Exception as exc:
            raise StoreUnreachable(f"{tool}: the call raised {type(exc).__name__}: {exc}") from exc
        if isinstance(reply, Mapping) and "error" in reply:
            raise StoreUnreachable(f"{tool}: {reply.get('error_type', 'error')}: {reply['error']}")
        try:
            return parse(_mapping(reply, "reply"))
        except _MalformedReply as exc:
            raise StoreUnreachable(f"{tool}: malformed reply: {exc}") from exc

    def find_tasks_by_metadata(self, project_root: str, *, key: str, value: str | int) -> tuple[TaskHit, ...]:
        hits: list[TaskHit] = []
        for _ in range(FIND_MAX_PAGES):
            page = self._call(
                "find_tasks_by_metadata",
                {"project_root": project_root, "key": key, "value": value, "limit": FIND_PAGE_SIZE, "offset": len(hits)},
                _parse_find_page,
            )
            hits.extend(page.hits)
            if not page.has_more:
                return tuple(hits)
        raise StoreUnreachable(f"find_tasks_by_metadata: still has_more after {FIND_MAX_PAGES} pages")

    def search_tasks(self, project_root: str, *, query: str, score_threshold: float) -> tuple[SearchHit, ...]:
        return self._call(
            "search_tasks",
            {"project_root": project_root, "query": query, "score_threshold": score_threshold},
            _parse_search,
        )

    def get_task(self, project_root: str, task_id: int) -> TaskRecord:
        return self._call("get_task", {"project_root": project_root, "id": str(task_id)}, _parse_task)

    def resolve_ticket(self, project_root: str, ticket: str, *, timeout_secs: float) -> TicketOutcome:
        return self._call(
            "resolve_ticket",
            {"project_root": project_root, "ticket": ticket, "timeout_seconds": timeout_secs},
            _parse_ticket,
            timeout=timeout_secs + RESOLVE_TRANSPORT_MARGIN_SECS,
        )
