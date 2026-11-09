#!/usr/bin/env python3
"""Whether a findings run has landed enough of its filed work for the next run.

``--run R --threshold T --project-root P`` weighs the tasks carrying
``x_finding_run=R`` by priority and prints a compact JSON verdict as its last
line. It exits 0 when the landed share reaches T, 75 (``os.EX_TEMPFAIL``) when
it has not yet, and 2 when it could not check. ``--pending`` reports the open
trigger chains. The measure is docs/quality-findings-contract.md §11; the
design is plans/completion-driven-triggers-prd.md C3.
"""

from __future__ import annotations

import enum
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, NamedTuple, TypeGuard

from dashboard.config import DEFAULT_FUSED_MEMORY_URLS
from shared.mcp_session import SyncMcpSession
from shared.task_statuses import TaskStatus

PRIORITY_WEIGHTS: Final[Mapping[str, int]] = MappingProxyType(
    {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'polish': 1}
)
MISSING_PRIORITY: Final = 'medium'
SHARE_DIGITS: Final = 4


class Verdict(enum.StrEnum):
    PASS = 'pass'
    NOT_YET = 'not_yet'
    ERROR = 'error'


class FailureCode(enum.StrEnum):
    """Why a check could not conclude; the ``error`` field of an error verdict."""

    EMPTY_RUN = 'empty_run'
    UNKNOWN_PRIORITY = 'unknown_priority'
    STORE_UNREACHABLE = 'store_unreachable'
    TOOL_ERROR = 'tool_error'
    MALFORMED_REPLY = 'malformed_reply'


class UnknownPriority(ValueError):
    def __init__(self, task_id: object, value: object) -> None:
        super().__init__(f'task {task_id} has priority {value!r}, not one of {", ".join(PRIORITY_WEIGHTS)}')
        self.task_id = task_id
        self.value = value


class TierCounts(NamedTuple):
    done: int
    open: int
    cancelled: int


def threshold_error(value: float) -> str | None:
    """None iff *value* is a usable threshold: a fraction in (0, 1], so never nan."""
    if 0 < value <= 1:
        return None
    return f'threshold {value!r} is outside (0, 1]'


@dataclass(frozen=True)
class CompletionVerdict:
    run_id: str
    threshold: float
    share: float | None
    landed: int
    pending: int
    total: int
    vacuous: bool
    passed: bool
    by_priority: Mapping[str, TierCounts]

    @property
    def verdict(self) -> Verdict:
        if self.passed:
            return Verdict.PASS
        return Verdict.ERROR if self.total == 0 else Verdict.NOT_YET

    @property
    def error(self) -> FailureCode | None:
        return FailureCode.EMPTY_RUN if self.total == 0 else None

    def to_dict(self) -> dict[str, Any]:
        rendered: dict[str, Any] = {
            'run': self.run_id,
            'verdict': self.verdict.value,
            'share': None if self.share is None else round(self.share, SHARE_DIGITS),
            'threshold': self.threshold,
            'landed': self.landed,
            'pending': self.pending,
            'total': self.total,
            'vacuous': self.vacuous,
            'by_priority': {tier: list(counts) for tier, counts in self.by_priority.items()},
        }
        if self.error is not None:
            rendered['error'] = self.error.value
        return rendered

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(',', ':'))


def _tier(row: Mapping[str, object]) -> str:
    value = row.get('priority')
    if value is None:
        return MISSING_PRIORITY
    if not isinstance(value, str) or value not in PRIORITY_WEIGHTS:
        raise UnknownPriority(row.get('task_id'), value)
    return value


def _slot(status: object) -> int:
    if status == TaskStatus.DONE:
        return 0
    return 2 if status == TaskStatus.CANCELLED else 1


def _tally(tasks: Iterable[Mapping[str, object]]) -> Mapping[str, TierCounts]:
    counts = {tier: [0, 0, 0] for tier in PRIORITY_WEIGHTS}
    for row in tasks:
        counts[_tier(row)][_slot(row.get('status'))] += 1
    return MappingProxyType({tier: TierCounts(*slots) for tier, slots in counts.items() if any(slots)})


def weighted_completion(
    tasks: Iterable[Mapping[str, object]], threshold: float, *, run_id: str,
) -> CompletionVerdict:
    """The priority-weighted landed share of *tasks*, judged against *threshold*."""
    problem = threshold_error(threshold)
    if problem is not None:
        raise ValueError(problem)
    by_priority = _tally(tasks)
    landed = sum(PRIORITY_WEIGHTS[tier] * counts.done for tier, counts in by_priority.items())
    pending = sum(PRIORITY_WEIGHTS[tier] * counts.open for tier, counts in by_priority.items())
    total = sum(sum(counts) for counts in by_priority.values())
    weighed = landed + pending
    vacuous = total > 0 and weighed == 0
    return CompletionVerdict(
        run_id=run_id,
        threshold=threshold,
        share=landed / weighed if weighed else None,
        landed=landed,
        pending=pending,
        total=total,
        vacuous=vacuous,
        passed=vacuous or (weighed > 0 and Fraction(landed, weighed) >= Fraction(str(threshold))),
        by_priority=by_priority,
    )


CallTool = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]

FUSED_MEMORY_URL_ENV: Final = 'FUSED_MEMORY_MCP_URL'
FINDING_RUN_KEY: Final = 'x_finding_run'
LOOKUP_TOOL: Final = 'find_tasks_by_metadata'
LOOKUP_PAGE_SIZE: Final = 100
MAX_LOOKUP_PAGES: Final = 200
CALL_TIMEOUT_SECS: Final = 30.0


def fused_memory_url(env: Mapping[str, str] = os.environ) -> str:
    return env.get(FUSED_MEMORY_URL_ENV, DEFAULT_FUSED_MEMORY_URLS[0])


def call_fused_memory_tool(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The default binding: one MCP session per call, at the URL configured now."""
    with SyncMcpSession(fused_memory_url(), client_name='check-run-completion', timeout=CALL_TIMEOUT_SECS) as session:
        return session.call_tool(name, arguments)


class LookupFailed(Exception):
    """The store answered nothing usable; never to be read as "no tasks"."""

    code: ClassVar[FailureCode]


class StoreUnreachable(LookupFailed):
    code = FailureCode.STORE_UNREACHABLE


class ToolErrorReply(LookupFailed):
    code = FailureCode.TOOL_ERROR


class MalformedReply(LookupFailed):
    code = FailureCode.MALFORMED_REPLY


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _call(call_tool: CallTool, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    where = f'{name} for {arguments.get("project_root")}'
    try:
        reply = call_tool(name, arguments)
    except Exception as exc:
        raise StoreUnreachable(f'{where}: {type(exc).__name__}: {exc}') from exc
    if not isinstance(reply, Mapping):
        raise MalformedReply(f'{where}: the reply is a {type(reply).__name__}, not an object')
    if 'error' in reply:
        raise ToolErrorReply(f'{where}: {reply.get("error_type", "error")}: {reply["error"]}')
    return reply


@dataclass(frozen=True)
class _LookupPage:
    rows: tuple[Mapping[str, Any], ...]
    has_more: bool
    tasks_with_key: int


def _lookup_row(row: object, where: str) -> Mapping[str, Any]:
    if not isinstance(row, Mapping) or not _is_int(row.get('task_id')) or not isinstance(row.get('status'), str):
        raise MalformedReply(f'{where}: a result row lacks an int task_id or a str status: {row!r:.200}')
    return row


def _lookup_page(reply: Mapping[str, Any], where: str) -> _LookupPage:
    results = reply.get('results')
    pagination = reply.get('pagination')
    tasks_with_key = reply.get('tasks_with_key')
    if not isinstance(results, list):
        raise MalformedReply(f'{where}: the reply carries no results list')
    if not isinstance(pagination, Mapping) or not isinstance(pagination.get('has_more'), bool):
        raise MalformedReply(f'{where}: the reply carries no bool pagination.has_more')
    if not _is_int(tasks_with_key):
        raise MalformedReply(f'{where}: tasks_with_key {tasks_with_key!r:.200} is not an int')
    if pagination['has_more'] and not results:
        raise MalformedReply(f'{where}: pagination.has_more is true on a page that returned nothing')
    rows = tuple(_lookup_row(row, where) for row in results)
    return _LookupPage(rows, pagination['has_more'], tasks_with_key)


def _lookup_pages(call_tool: CallTool, arguments: Mapping[str, Any]) -> tuple[tuple[Mapping[str, Any], ...], int]:
    """Every row the lookup matches, paged by the returned count, and the first page's tasks_with_key."""
    where = f'{LOOKUP_TOOL} for {arguments.get("project_root")}'
    rows: list[Mapping[str, Any]] = []
    tasks_with_key = 0
    for page_number in range(MAX_LOOKUP_PAGES):
        page = _lookup_page(_call(call_tool, LOOKUP_TOOL, {**arguments, 'offset': len(rows)}), where)
        if page_number == 0:
            tasks_with_key = page.tasks_with_key
        rows.extend(page.rows)
        if not page.has_more:
            return tuple(rows), tasks_with_key
    raise MalformedReply(f'{where}: still has_more after {MAX_LOOKUP_PAGES} pages')


def _absolute_root(project_root: str) -> str:
    return str(Path(project_root).resolve())


@dataclass(frozen=True)
class RunSelection:
    rows: tuple[Mapping[str, Any], ...]
    tasks_with_key: int


def run_tasks(project_root: str, run_id: str, *, call_tool: CallTool = call_fused_memory_tool) -> RunSelection:
    """Every task carrying ``x_finding_run=run_id``, in any status."""
    arguments = {
        'project_root': _absolute_root(project_root),
        'key': FINDING_RUN_KEY,
        'value': run_id,
        'limit': LOOKUP_PAGE_SIZE,
    }
    rows, tasks_with_key = _lookup_pages(call_tool, arguments)
    return RunSelection(rows, tasks_with_key)
