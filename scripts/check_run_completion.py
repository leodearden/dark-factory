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
import math
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, NamedTuple, NoReturn, TypeGuard

from _task_db_scan import discover_project_roots
from dashboard.config import DEFAULT_FUSED_MEMORY_URLS
from orchestrator.predicate_recheck import PredicateRecheck
from pydantic import ValidationError
from shared.mcp_session import McpCallError, McpFailureKind, SyncMcpSession
from shared.task_metadata import TRIGGER_CHAIN_KEY, TriggerChain
from shared.task_statuses import ACTIVE, TaskStatus

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


def _shown_share(share: Fraction) -> float:
    """*share* cut, never rounded up, to SHARE_DIGITS: it prints as reaching a threshold only if it does."""
    scale = 10**SHARE_DIGITS
    return math.floor(share * scale) / scale


@dataclass(frozen=True)
class CompletionVerdict:
    """A run's tier counts judged against a threshold; every other figure derives from those, so none disagree."""

    run_id: str
    threshold: float
    by_priority: Mapping[str, TierCounts]

    def __post_init__(self) -> None:
        problem = threshold_error(self.threshold)
        if problem is not None:
            raise ValueError(problem)
        unknown = [tier for tier in self.by_priority if tier not in PRIORITY_WEIGHTS]
        if unknown:
            raise ValueError(f'tiers {unknown} are not among {", ".join(PRIORITY_WEIGHTS)}')
        if any(count < 0 for counts in self.by_priority.values() for count in counts):
            raise ValueError(f'a tier count is negative: {dict(self.by_priority)}')

    @property
    def landed(self) -> int:
        return sum(PRIORITY_WEIGHTS[tier] * counts.done for tier, counts in self.by_priority.items())

    @property
    def pending(self) -> int:
        return sum(PRIORITY_WEIGHTS[tier] * counts.open for tier, counts in self.by_priority.items())

    @property
    def total(self) -> int:
        return sum(sum(counts) for counts in self.by_priority.values())

    @property
    def share(self) -> Fraction | None:
        """The exact landed share of the weighed tasks; None when nothing is weighed."""
        weighed = self.landed + self.pending
        return Fraction(self.landed, weighed) if weighed else None

    @property
    def vacuous(self) -> bool:
        return self.total > 0 and self.share is None

    @property
    def passed(self) -> bool:
        share = self.share
        return self.vacuous if share is None else share >= Fraction(str(self.threshold))

    @property
    def verdict(self) -> Verdict:
        if self.passed:
            return Verdict.PASS
        return Verdict.ERROR if self.total == 0 else Verdict.NOT_YET

    @property
    def error(self) -> FailureCode | None:
        return FailureCode.EMPTY_RUN if self.total == 0 else None

    def to_dict(self) -> dict[str, Any]:
        share = self.share
        return _verdict_record(
            self.run_id,
            self.verdict,
            self.threshold,
            self.error,
            share=None if share is None else _shown_share(share),
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
    return CompletionVerdict(run_id=run_id, threshold=threshold, by_priority=_tally(tasks))


CallTool = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]

FUSED_MEMORY_URL_ENV: Final = 'FUSED_MEMORY_MCP_URL'
FINDING_RUN_KEY: Final = 'x_finding_run'
LOOKUP_TOOL: Final = 'find_tasks_by_metadata'
LOOKUP_PAGE_SIZE: Final = 100
MAX_LOOKUP_PAGES: Final = 200
CALL_TIMEOUT_SECS: Final = 30.0


class LookupFailed(Exception):
    """The store answered nothing usable; never to be read as "no tasks"."""

    code: ClassVar[FailureCode]


class StoreUnreachable(LookupFailed):
    code = FailureCode.STORE_UNREACHABLE


class ToolErrorReply(LookupFailed):
    code = FailureCode.TOOL_ERROR


class MalformedReply(LookupFailed):
    code = FailureCode.MALFORMED_REPLY


FAILURE_BY_MCP_KIND: Final[Mapping[McpFailureKind, type[LookupFailed]]] = MappingProxyType({
    McpFailureKind.TRANSPORT: StoreUnreachable,
    McpFailureKind.HTTP_STATUS: StoreUnreachable,
    McpFailureKind.TOOL_ERROR: ToolErrorReply,
    McpFailureKind.JSON_RPC_ERROR: ToolErrorReply,
    McpFailureKind.REPLY_SHAPE: MalformedReply,
    McpFailureKind.UNDECODABLE: MalformedReply,
})


def _call_site(name: str, arguments: Mapping[str, Any]) -> str:
    return f'{name} for {arguments.get("project_root")}'


def fused_memory_url(env: Mapping[str, str] = os.environ) -> str:
    return env.get(FUSED_MEMORY_URL_ENV, DEFAULT_FUSED_MEMORY_URLS[0])


def call_fused_memory_tool(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The default binding: one MCP session per call, at the URL configured now.

    An MCP failure is raised as the LookupFailed its kind names, so a refusal never reads as unreachable.
    """
    try:
        with SyncMcpSession(
            fused_memory_url(), client_name='check-run-completion', timeout=CALL_TIMEOUT_SECS,
        ) as session:
            return session.call_tool(name, arguments)
    except McpCallError as exc:
        raise FAILURE_BY_MCP_KIND[exc.kind](f'{_call_site(name, arguments)}: {exc}') from exc


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _call(call_tool: CallTool, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    """*call_tool*'s reply, checked; a binding's LookupFailed passes through, any other raise is unreachable."""
    where = _call_site(name, arguments)
    try:
        reply = call_tool(name, arguments)
    except LookupFailed:
        raise
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
    where = _call_site(LOOKUP_TOOL, arguments)
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
    """Raises UsageError instead of exiting, and -h/--help is a plain flag, so parsing never prints or exits."""

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
    parser = _Parser(
        prog=PROG, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False,
    )
    parser.add_argument('-h', '--help', action='store_true', help='show this help on stderr; runs no check and exits 2')
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
    if args.help:
        raise UsageError('-h/--help prints this text and runs no check', parser.format_help())
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


GET_TASK_TOOL: Final = 'get_task'


class ProjectState(enum.StrEnum):
    NEVER_FILED = 'never_filed'
    NO_OPEN_CHAINS = 'no_open_chains'
    OPEN_CHAINS = 'open_chains'
    ERROR = 'error'


@dataclass(frozen=True)
class CompletionGateReport:
    task_id: int
    status: str
    next_check_at: str | None
    streak: int
    threshold: float | None
    threshold_unreadable: str | None

    def to_dict(self) -> dict[str, Any]:
        rendered: dict[str, Any] = {
            'id': self.task_id,
            'status': self.status,
            'next_check_at': self.next_check_at,
            'streak': self.streak,
            'threshold': self.threshold,
        }
        if self.threshold_unreadable is not None:
            rendered['threshold_unreadable'] = self.threshold_unreadable
        return rendered


@dataclass(frozen=True)
class HumanGateReport:
    task_id: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {'id': self.task_id, 'status': self.status}


@dataclass(frozen=True)
class ChainReport:
    skill: str
    run_id: str
    completion_gate: CompletionGateReport | None
    human_gate: HumanGateReport | None
    verdict: CompletionVerdict | CheckFailure | None

    def to_dict(self) -> dict[str, Any]:
        return {
            'skill': self.skill,
            'run_id': self.run_id,
            'completion_gate': None if self.completion_gate is None else self.completion_gate.to_dict(),
            'human_gate': None if self.human_gate is None else self.human_gate.to_dict(),
            'verdict': None if self.verdict is None else self.verdict.to_dict(),
        }


@dataclass(frozen=True)
class ProjectReport:
    project_root: str
    state: ProjectState
    chains: tuple[ChainReport, ...] = ()
    unreadable_task_ids: tuple[int, ...] = ()
    chain_tasks: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        rendered: dict[str, Any] = {'project_root': self.project_root, 'state': self.state.value}
        if self.error is not None:
            rendered['error'] = self.error
        rendered['unreadable_task_ids'] = list(self.unreadable_task_ids)
        rendered['chains'] = [chain.to_dict() for chain in self.chains]
        return rendered


@dataclass(frozen=True)
class _ChainLink:
    row: Mapping[str, Any]
    link: TriggerChain


def _get_task(call_tool: CallTool, project_root: str, task_id: int) -> tuple[str, Mapping[str, Any]]:
    """A task's status and metadata, under the same reply checks as the lookup."""
    reply = _call(call_tool, GET_TASK_TOOL, {'id': str(task_id), 'project_root': project_root})
    status, metadata = reply.get('status'), reply.get('metadata')
    if not isinstance(status, str) or not isinstance(metadata, Mapping):
        raise MalformedReply(f'{GET_TASK_TOOL} for task {task_id} in {project_root}: no str status and metadata object')
    return status, metadata


def _whole_chain_value(call_tool: CallTool, project_root: str, row: Mapping[str, Any]) -> object:
    """The row's whole trigger_chain; one the lookup's byte budget elided is re-read with get_task."""
    if row.get('truncated_value') is not True:
        return row.get('matched_value')
    _, metadata = _get_task(call_tool, project_root, row['task_id'])
    return metadata.get(TRIGGER_CHAIN_KEY)


def _chain_links(
    call_tool: CallTool, project_root: str, rows: Iterable[Mapping[str, Any]],
) -> tuple[tuple[_ChainLink, ...], tuple[int, ...]]:
    links: list[_ChainLink] = []
    unreadable: list[int] = []
    for row in rows:
        value = _whole_chain_value(call_tool, project_root, row)
        try:
            links.append(_ChainLink(row, TriggerChain.model_validate(value)))
        except (ValidationError, TypeError):
            if row['status'] in ACTIVE:
                unreadable.append(row['task_id'])
    return tuple(links), tuple(unreadable)


def _open_chains(links: Iterable[_ChainLink]) -> dict[tuple[str, str], dict[str, Mapping[str, Any]]]:
    """Each open chain's row per role, keyed by (skill, run_id).

    A chain is open while one of its unsuperseded rows is ACTIVE (the C2 guard's rule). Per
    role an ACTIVE row wins, so a chain re-filed for the same run reports its live gates; a
    terminal row stays attached when no live row shares its role, so a passed completion
    gate still reports under its blocked human gate rather than reading as the interim form.
    """
    chains: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for item in sorted(links, key=lambda item: item.row['status'] not in ACTIVE):
        if item.link.superseded_by is None:
            chains.setdefault((item.link.skill, item.link.run_id), {}).setdefault(item.link.role, item.row)
    return {
        chain: roles for chain, roles in chains.items()
        if any(row['status'] in ACTIVE for row in roles.values())
    }


def _milestone_at(metadata: Mapping[str, Any]) -> str | None:
    milestone = metadata.get('milestone')
    at = milestone.get('at') if isinstance(milestone, Mapping) else None
    return at if isinstance(at, str) else None


def _gate_request(metadata: Mapping[str, Any]) -> CheckRequest | str:
    """The check the gate's predicate runs (D9), or why its args cannot be read back."""
    before_done = metadata.get('before_done')
    args = before_done.get('args') if isinstance(before_done, Mapping) else None
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        return 'the gate carries no before_done.args list of strings'
    try:
        request = parse_args(args)
    except UsageError as exc:
        return str(exc)
    return request if isinstance(request, CheckRequest) else 'before_done.args ask for --pending, not a check'


def _completion_gate(
    call_tool: CallTool, project_root: str, task_id: int,
) -> tuple[CompletionGateReport, CheckRequest | None]:
    status, metadata = _get_task(call_tool, project_root, task_id)
    request = _gate_request(metadata)
    readable = request if isinstance(request, CheckRequest) else None
    recheck = PredicateRecheck.from_metadata(metadata)
    report = CompletionGateReport(
        task_id=task_id,
        status=status,
        next_check_at=_milestone_at(metadata),
        streak=0 if recheck is None else recheck.consecutive,
        threshold=None if readable is None else readable.threshold,
        threshold_unreadable=request if isinstance(request, str) else None,
    )
    return report, readable


def _chain_report(
    call_tool: CallTool, project_root: str, chain: tuple[str, str], rows: Mapping[str, Mapping[str, Any]],
) -> ChainReport:
    skill, run_id = chain
    human_row = rows.get('human_gate')
    human = None if human_row is None else HumanGateReport(human_row['task_id'], human_row['status'])
    gate_row = rows.get('completion_gate')
    if gate_row is None:
        return ChainReport(skill, run_id, None, human, None)
    gate, request = _completion_gate(call_tool, project_root, gate_row['task_id'])
    verdict = None if request is None else evaluate(request, call_tool=call_tool).result
    return ChainReport(skill, run_id, gate, human, verdict)


def _read_project(call_tool: CallTool, project_root: str) -> ProjectReport:
    arguments = {
        'project_root': project_root,
        'key': TRIGGER_CHAIN_KEY,
        'include_value': True,
        'limit': LOOKUP_PAGE_SIZE,
    }
    rows, _ = _lookup_pages(call_tool, arguments)
    if not rows:
        return ProjectReport(project_root, ProjectState.NEVER_FILED)
    links, unreadable = _chain_links(call_tool, project_root, rows)
    chains = tuple(
        _chain_report(call_tool, project_root, chain, roles) for chain, roles in _open_chains(links).items()
    )
    state = ProjectState.OPEN_CHAINS if chains else ProjectState.NO_OPEN_CHAINS
    return ProjectReport(project_root, state, chains, unreadable, chain_tasks=len(rows))


def read_pending(
    project_roots: Iterable[str], *, call_tool: CallTool = call_fused_memory_tool,
) -> tuple[ProjectReport, ...]:
    """One report per root; a root whose store read fails is reported as an error, never dropped."""
    reports: list[ProjectReport] = []
    for root in map(_absolute_root, project_roots):
        try:
            reports.append(_read_project(call_tool, root))
        except LookupFailed as exc:
            reports.append(ProjectReport(root, ProjectState.ERROR, error=str(exc)))
    return tuple(reports)


def _gate_line(gate: CompletionGateReport | None) -> str:
    if gate is None:
        return 'completion gate: none (interim form), no threshold or share measured'
    threshold = gate.threshold if gate.threshold_unreadable is None else f'unreadable: {gate.threshold_unreadable}'
    return (
        f'completion gate {gate.task_id}: {gate.status}, next check {gate.next_check_at or "unscheduled"}, '
        f'streak {gate.streak}, threshold {threshold}'
    )


def _share_line(verdict: CompletionVerdict | CheckFailure) -> str:
    rendered = verdict.to_dict()
    share = 'none' if rendered['share'] is None else rendered['share']
    outcome = 'pass, vacuous' if rendered['vacuous'] else rendered.get('error', rendered['verdict'])
    detail = f': {verdict.detail}' if isinstance(verdict, CheckFailure) else ''
    return f'share {share} ({outcome}){detail}'


def _chain_lines(chain: ChainReport) -> list[str]:
    human = 'human gate: none' if chain.human_gate is None else (
        f'human gate {chain.human_gate.task_id}: {chain.human_gate.status}'
    )
    lines = [_gate_line(chain.completion_gate), human]
    if chain.verdict is not None:
        lines.append(_share_line(chain.verdict))
    return [f'{chain.skill} {chain.run_id}', *(f'  {line}' for line in lines)]


def _project_lines(report: ProjectReport) -> list[str]:
    root = report.project_root
    if report.state is ProjectState.ERROR:
        return [f'project {root}: could not read: {report.error}']
    if report.state is ProjectState.NEVER_FILED:
        body = [f'no trigger chain has been filed in {root}']
    elif report.state is ProjectState.NO_OPEN_CHAINS:
        body = [f'no open trigger chain in {root} ({report.chain_tasks} chain tasks, none open)']
    else:
        body = [line for chain in report.chains for line in _chain_lines(chain)]
    body += [f'unreadable trigger_chain on task {task_id}' for task_id in report.unreadable_task_ids]
    return [f'project {root}', *(f'  {line}' for line in body)]


def render_text(reports: Sequence[ProjectReport]) -> list[str]:
    if not reports:
        return ['no project root to report: none named and none discovered']
    return [line for report in reports for line in _project_lines(report)]


def render_json(reports: Sequence[ProjectReport]) -> str:
    return json.dumps({'projects': [report.to_dict() for report in reports]})


def _report_pending(request: PendingRequest) -> int:
    reports = read_pending(request.project_roots or discover_project_roots())
    if request.as_json:
        print(render_json(reports))
    else:
        print('\n'.join(render_text(reports)))
    return COULD_NOT_CHECK if any(report.state is ProjectState.ERROR for report in reports) else os.EX_OK


def main(argv: Sequence[str] | None = None) -> int:
    try:
        request = parse_args(sys.argv[1:] if argv is None else argv)
    except UsageError as exc:
        print(f'{exc.usage}{PROG}: error: {exc}', file=sys.stderr)
        return COULD_NOT_CHECK
    if isinstance(request, PendingRequest):
        return _report_pending(request)
    outcome = evaluate(request)
    for line in human_lines(outcome):
        print(line)
    print(outcome.result.to_json(), flush=True)
    return exit_code(outcome.result)


if __name__ == '__main__':
    sys.exit(main())
