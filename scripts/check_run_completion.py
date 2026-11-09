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

import argparse
import enum
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, NamedTuple, NoReturn, TypeGuard

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


def _verdict_record(
    run_id: str,
    verdict: Verdict,
    threshold: float,
    error: FailureCode | None,
    *,
    share: float | None = None,
    landed: int | None = None,
    pending: int | None = None,
    total: int | None = None,
    vacuous: bool | None = None,
    by_priority: Mapping[str, list[int]] | None = None,
) -> dict[str, Any]:
    """The one key order of a verdict's JSON; a value never measured is null."""
    record: dict[str, Any] = {
        'run': run_id,
        'verdict': verdict.value,
        'share': share,
        'threshold': threshold,
        'landed': landed,
        'pending': pending,
        'total': total,
        'vacuous': vacuous,
        'by_priority': by_priority,
    }
    if error is not None:
        record['error'] = error.value
    return record


def _compact_json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, separators=(',', ':'))


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
        return _verdict_record(
            self.run_id,
            self.verdict,
            self.threshold,
            self.error,
            share=None if self.share is None else round(self.share, SHARE_DIGITS),
            landed=self.landed,
            pending=self.pending,
            total=self.total,
            vacuous=self.vacuous,
            by_priority={tier: list(counts) for tier, counts in self.by_priority.items()},
        )

    def to_json(self) -> str:
        return _compact_json(self.to_dict())


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


PROG: Final = Path(__file__).name
COULD_NOT_CHECK: Final = 2


class UsageError(Exception):
    def __init__(self, message: str, usage: str) -> None:
        super().__init__(message)
        self.usage = usage


class _Parser(argparse.ArgumentParser):
    """Raises UsageError instead of exiting, so a gate's args can be read back safely."""

    def error(self, message: str) -> NoReturn:
        raise UsageError(message, self.format_usage())


def _threshold(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f'threshold {text!r} is not a number') from None
    problem = threshold_error(value)
    if problem is not None:
        raise argparse.ArgumentTypeError(problem)
    return value


def _parser() -> _Parser:
    parser = _Parser(prog=PROG, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', help='the run id the tasks carry as x_finding_run')
    parser.add_argument('--threshold', type=_threshold, help='the landed share that passes, in (0, 1]')
    parser.add_argument('--project-root', action='append', type=_absolute_root, help='the project to read')
    parser.add_argument('--pending', action='store_true', help='report the open trigger chains instead')
    parser.add_argument('--json', action='store_true', dest='as_json', help='--pending only: one JSON document')
    return parser


@dataclass(frozen=True)
class CheckRequest:
    run_id: str
    threshold: float
    project_root: str

    def __post_init__(self) -> None:
        problem = threshold_error(self.threshold)
        if problem is not None:
            raise ValueError(problem)


@dataclass(frozen=True)
class PendingRequest:
    project_roots: tuple[str, ...]
    as_json: bool


def parse_args(argv: Sequence[str]) -> CheckRequest | PendingRequest:
    parser = _parser()
    args = parser.parse_args(list(argv))
    roots: list[str] = args.project_root or []
    if args.pending:
        if args.run is not None or args.threshold is not None:
            parser.error('--pending reports every open chain and takes no --run or --threshold')
        return PendingRequest(tuple(roots), args.as_json)
    if args.run is None or args.threshold is None or len(roots) != 1:
        parser.error('a check needs --run, --threshold and exactly one --project-root')
    if args.as_json:
        parser.error('--json applies to --pending; a check always ends with its JSON verdict')
    return CheckRequest(args.run, args.threshold, roots[0])


@dataclass(frozen=True)
class CheckFailure:
    """A check that could not conclude; *detail* is for the human lines only."""

    run_id: str
    threshold: float
    code: FailureCode
    detail: str

    @property
    def verdict(self) -> Verdict:
        return Verdict.ERROR

    def to_dict(self) -> dict[str, Any]:
        return _verdict_record(self.run_id, self.verdict, self.threshold, self.code)

    def to_json(self) -> str:
        return _compact_json(self.to_dict())


@dataclass(frozen=True)
class CheckOutcome:
    """A check's result, with the lookup's tasks_with_key (None when the lookup failed)."""

    request: CheckRequest
    result: CompletionVerdict | CheckFailure
    tasks_with_key: int | None


def evaluate(request: CheckRequest, *, call_tool: CallTool = call_fused_memory_tool) -> CheckOutcome:
    """The gate's predicate: select the run's tasks, then weigh them. Never raises a store failure."""
    try:
        selection = run_tasks(request.project_root, request.run_id, call_tool=call_tool)
    except LookupFailed as exc:
        return CheckOutcome(request, CheckFailure(request.run_id, request.threshold, exc.code, str(exc)), None)
    try:
        verdict = weighted_completion(selection.rows, request.threshold, run_id=request.run_id)
    except UnknownPriority as exc:
        failure = CheckFailure(request.run_id, request.threshold, FailureCode.UNKNOWN_PRIORITY, str(exc))
        return CheckOutcome(request, failure, selection.tasks_with_key)
    return CheckOutcome(request, verdict, selection.tasks_with_key)


def _empty_run_line(outcome: CheckOutcome) -> str:
    root = outcome.request.project_root
    if not outcome.tasks_with_key:
        return (
            f'no task in {root} carries {FINDING_RUN_KEY} at all '
            '(no instrument run has filed under the contract yet)'
        )
    return (
        f'{FINDING_RUN_KEY} is live in {root} ({outcome.tasks_with_key} tasks) '
        f'but no task carries {FINDING_RUN_KEY}={outcome.request.run_id}'
    )


def _judgement(verdict: CompletionVerdict) -> str:
    if verdict.vacuous:
        return 'vacuous pass: every task is cancelled, nothing is left to land'
    share = verdict.to_dict()['share']
    return f'share {share} against threshold {verdict.threshold}: {"pass" if verdict.passed else "not yet"}'


def human_lines(outcome: CheckOutcome) -> list[str]:
    result = outcome.result
    header = f'run {outcome.request.run_id} in {outcome.request.project_root}'
    if isinstance(result, CheckFailure):
        return [f'{header}: could not check ({result.code.value}): {result.detail}']
    if result.error is FailureCode.EMPTY_RUN:
        return [f'{header}: could not check ({FailureCode.EMPTY_RUN.value})', _empty_run_line(outcome)]
    return [
        f'{header}: {result.total} tasks',
        f'landed weight {result.landed}, pending weight {result.pending}',
        _judgement(result),
    ]


def exit_code(result: CompletionVerdict | CheckFailure) -> int:
    if result.verdict is Verdict.PASS:
        return os.EX_OK
    return os.EX_TEMPFAIL if result.verdict is Verdict.NOT_YET else COULD_NOT_CHECK


def main(argv: Sequence[str] | None = None) -> int:
    try:
        request = parse_args(sys.argv[1:] if argv is None else argv)
    except UsageError as exc:
        print(f'{exc.usage}{PROG}: error: {exc}', file=sys.stderr)
        return COULD_NOT_CHECK
    if isinstance(request, PendingRequest):
        raise NotImplementedError('--pending lands with the report layer')
    outcome = evaluate(request)
    for line in human_lines(outcome):
        print(line)
    print(outcome.result.to_json(), flush=True)
    return exit_code(outcome.result)


if __name__ == '__main__':
    sys.exit(main())
