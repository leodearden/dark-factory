"""The client defect filer's pass (plans/client-feed-prd.md §14.4, §14.5): each portal defect report filed once in its project as a fix and the human gate it waits on, re-derived from the store every pass."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError
from shared.defects import DEFECT_TRANSITIONS, DefectStatus, DefectTrigger

from client_feed.contract import DefectReported, DefectRuled, TaskStatus
from client_feed.defect_ref import check_ref, portal_ref
from client_feed.filer_state import FilerCursor, RefRecord, StateDir, held_state_dir
from client_feed.filer_tasks import (
    ClientDefectFacts,
    FilerFallback,
    facts_of,
    fix_submission,
    gate_submission,
    is_filer_withdrawal,
    operator_acceptance_event,
    requires_gate,
    withdrawal_event,
)
from client_feed.filer_tools import (
    FILER_IDENTITY,
    ClosedTools,
    Disposition,
    ListingCaller,
    OpenSession,
    OpenTools,
    RefusalKind,
    SessionUnavailable,
    TaskView,
    ToolRefused,
    WithheldWrites,
    closed_tools,
    missing_tools,
    opened,
)
from client_feed.inbox import FeedIoProblem, InboxRead, InboxUnusable, read_inbox
from client_feed.policy import FilerPolicy, Policy

logger = logging.getLogger(__name__)

FALLBACK_AFTER_ATTEMPTS: Final = 5
MAX_BACKOFF_SECONDS: Final = 300

_DEFERRED: Final = TaskStatus.DEFERRED
_TERMINAL: Final = frozenset({TaskStatus.DONE, TaskStatus.CANCELLED})
_CANCELLABLE_FIX: Final = frozenset({TaskStatus.PENDING, TaskStatus.DEFERRED})
_WITHDRAWABLE: Final = frozenset(
    status for (status, trigger) in DEFECT_TRANSITIONS if trigger is DefectTrigger.DECLARE_CLOSURE and status is not None
)


class Unconverged(BaseModel):
    """A ref the pass could not bring to its final state, and the refusal that stopped it."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    ref: str
    tool: str
    kind: RefusalKind
    code: str
    disposition: Disposition
    attempts: int


class PassReport(BaseModel):
    """What one pass read and did; it names refusals by kind and code only."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    offset_from: int
    offset_to: int
    converged: tuple[str, ...]
    unconverged: tuple[Unconverged, ...]
    rejected_lines: int
    inbox_problem: str | None = None


@dataclass(frozen=True)
class _RefState:
    """A ref's final state over the lines read: its report, if read, and whether any ruling names it."""

    report: DefectReported | None
    ruled: bool


@dataclass(frozen=True)
class _Outcome:
    """A ref's result: converged, or the refusal that stopped it and the record the cursor keeps for it."""

    refused: Unconverged | None = None
    record: RefRecord | None = None


class _Held(Exception):
    """A refusal that leaves a ref unconverged, with the record the cursor keeps for the ref after it."""

    def __init__(self, refused: ToolRefused, record: RefRecord | None) -> None:
        self.refused = refused
        self.record = record
        super().__init__(str(refused))


@contextmanager
def _holding(record: RefRecord | None) -> Iterator[None]:
    try:
        yield
    except ToolRefused as refused:
        raise _Held(refused, record) from None


@dataclass(frozen=True)
class _Filed:
    """A ref's filer fix, the facts it was filed with, and the record its submission left."""

    fix: TaskView
    facts: ClientDefectFacts
    record: RefRecord | None


def _ref_states(read: InboxRead, project_id: str) -> dict[str, _RefState]:
    states: dict[str, _RefState] = {}
    for line in read.lines:
        if not isinstance(line, DefectReported | DefectRuled):
            continue
        ref = portal_ref(project_id, line.block, line.report)
        held = states.get(ref, _RefState(report=None, ruled=False))
        if isinstance(line, DefectRuled):
            states[ref] = _RefState(report=held.report, ruled=True)
        else:
            states[ref] = _RefState(report=held.report or line, ruled=held.ruled)
    return states


def _id_order(task_id: str) -> tuple[int, str]:
    return (int(task_id), '') if task_id.isdecimal() else (-1, task_id)


def _lowest(views: Sequence[TaskView], what: str, ref: str) -> TaskView | None:
    if len(views) > 1:
        logger.warning(
            '%s has %d filer %ss (%s); using the lowest id',
            ref,
            len(views),
            what,
            ', '.join(view.task_id for view in views),
        )
    return min(views, key=lambda view: _id_order(view.task_id), default=None)


def _filer_tasks(tools: ClosedTools, task_ids: Sequence[str]) -> list[TaskView]:
    return [view for view in map(tools.get_task, task_ids) if view.source == FILER_IDENTITY]


def _just_submitted(task_id: str) -> TaskView:
    """A fix this pass submitted, as the store now holds it, without reading it back (a --check plan's id names no task)."""
    return TaskView(
        task_id=task_id,
        status=_DEFERRED,
        dependencies=frozenset(),
        source=FILER_IDENTITY,
        x_client_defect=None,
        defects=(),
    )


def _facts(fix: TaskView, report: DefectReported, fallback: FilerFallback | None) -> ClientDefectFacts:
    try:
        return ClientDefectFacts.model_validate(fix.x_client_defect)
    except ValidationError:
        logger.warning('fix %s carries no readable x_client_defect; using its report line', fix.task_id)
        return facts_of(report, fallback)


def _gate_and_commit(
    tools: ClosedTools,
    report: DefectReported,
    ref: str,
    policy: Policy,
    filer: FilerPolicy,
    fix: TaskView,
    facts: ClientDefectFacts,
) -> None:
    """Gate a fix still held deferred, then release it; a released fix is never gated after the fact."""
    if fix.status is not _DEFERRED:
        return
    gate = _lowest(_filer_tasks(tools, tools.find_gates(ref)), 'gate', ref)
    gate_id, gate_status = (None, None) if gate is None else (gate.task_id, gate.status)
    if gate_id is None and requires_gate(facts, auto_dispatch=filer.auto_dispatch):
        submission = gate_submission(
            report,
            ref=ref,
            fix_task_id=fix.task_id,
            facts=facts,
            priority=filer.fix_priority,
            project_id=policy.project_id,
        )
        gate_id, gate_status = tools.submit_planned(submission), _DEFERRED
    if gate_id is not None and gate_id not in fix.dependencies:
        tools.add_dependency(fix.task_id, gate_id)
    held = [fix.task_id] + ([gate_id] if gate_id is not None and gate_status is _DEFERRED else [])
    tools.commit_planning(held)


def _after_refusal(record: RefRecord | None, refused: ToolRefused) -> RefRecord | None:
    """The record a refused restated fix leaves: a counted attempt, and the fallback once one applies."""
    if refused.disposition not in (Disposition.FALLBACK, Disposition.RETRY_THEN_FALLBACK):
        return record
    attempts = 1 if record is None else record.attempts + 1
    falls_back = refused.disposition is Disposition.FALLBACK or attempts >= FALLBACK_AFTER_ATTEMPTS
    fallback = FilerFallback(kind=refused.kind, code=refused.code) if falls_back else None
    return RefRecord(attempts=attempts, last_code=refused.code, fallback=fallback)


def _submitted(
    tools: ClosedTools, report: DefectReported, ref: str, filer: FilerPolicy, record: RefRecord | None
) -> _Filed:
    fallback = None if record is None else record.fallback
    with _holding(record):
        submission = fix_submission(report, ref=ref, priority=filer.fix_priority, fallback=fallback)
        task_id = tools.submit_planned(submission)
    return _Filed(_just_submitted(task_id), facts_of(report, fallback), record)


def _submit_fix(
    tools: ClosedTools, report: DefectReported, ref: str, filer: FilerPolicy, record: RefRecord | None
) -> _Filed:
    """Submit the fix; a refused restatement falls back to typed facts once its refusal says so, and only once."""
    if record is not None and record.fallback is not None:
        return _submitted(tools, report, ref, filer, record)
    try:
        return _submitted(tools, report, ref, filer, record)
    except _Held as held:
        record = _after_refusal(record, held.refused)
        if record is None or record.fallback is None:
            raise _Held(held.refused, record) from None
    return _submitted(tools, report, ref, filer, record)


def _filed_fix(
    tools: ClosedTools, report: DefectReported, ref: str, filer: FilerPolicy, record: RefRecord | None
) -> _Filed | None:
    """The ref's filer fix, submitted if the store holds none; None when the ref was filed by hand."""
    with _holding(record):
        carriers = [tools.get_task(carrier.task_id) for carrier in tools.lookup_defect_ref(ref)]
    mine = [view for view in carriers if view.source == FILER_IDENTITY]
    if carriers and not mine:
        return None
    fix = _lowest(mine, 'fix', ref)
    if fix is None:
        return _submit_fix(tools, report, ref, filer, record)
    return _Filed(fix, _facts(fix, report, None if record is None else record.fallback), record)


def converge_filing(
    tools: ClosedTools, report: DefectReported, ref: str, policy: Policy, record: RefRecord | None
) -> _Outcome:
    """File `ref` as a fix and, unless auto-dispatched clean, its gate; every step re-derived from the store."""
    filer = _filer(policy)
    try:
        filed = _filed_fix(tools, report, ref, filer, record)
        if filed is not None:
            with _holding(filed.record):
                _gate_and_commit(tools, report, ref, policy, filer, filed.fix, filed.facts)
    except _Held as held:
        return _Outcome(refused=_unconverged(ref, held.refused, held.record), record=held.record)
    return _Outcome()


def _unconverged(ref: str, refused: ToolRefused, record: RefRecord | None) -> Unconverged:
    return Unconverged(
        ref=ref,
        tool=refused.tool,
        kind=refused.kind,
        code=refused.code,
        disposition=refused.disposition,
        attempts=0 if record is None else record.attempts,
    )


def _pending_on_filer_withdrawal(fix: TaskView, ref: str) -> bool:
    return any(entry.get('ref') == ref and is_filer_withdrawal(entry) for entry in fix.defects)


def _warn_out_of_reach(ref: str, fix: TaskView, defect_status: DefectStatus) -> None:
    """A defect no closure may leave (proven, exempted, pending on another closure) takes no write; a withdrawn one is converged."""
    if defect_status is not DefectStatus.WITHDRAWN:
        logger.warning(
            "%s: the defect on fix %s is %s; the ruling withdraws nothing and cancels the filer's tasks",
            ref,
            fix.task_id,
            defect_status,
        )


def _withdraw(tools: ClosedTools, ref: str, fix: TaskView, defect_status: DefectStatus) -> bool:
    """Withdraw the fix's defect as not a defect and record Leo's acceptance; False once the store says the fix went terminal."""
    withdraws = defect_status in _WITHDRAWABLE
    accepts = withdraws or (defect_status is DefectStatus.PENDING and _pending_on_filer_withdrawal(fix, ref))
    if not accepts:
        _warn_out_of_reach(ref, fix, defect_status)
    try:
        if withdraws:
            tools.declare_withdrawal(fix.task_id, withdrawal_event(ref))
        if accepts:
            tools.record_acceptance(fix.task_id, ref, operator_acceptance_event(ref))
    except ToolRefused as refused:
        if refused.disposition is not Disposition.ALREADY_DONE:
            raise
        return False
    return True


def _withdraw_and_cancel_fix(tools: ClosedTools, ref: str) -> None:
    """The filer's fix of a ruled ref: its defect withdrawn while the fix is live, and the fix cancelled unless started."""
    carriers = {carrier.task_id: carrier for carrier in tools.lookup_defect_ref(ref)}
    fix = _lowest(_filer_tasks(tools, tuple(carriers)), 'fix', ref)
    if fix is None or fix.status in _TERMINAL:
        return
    if _withdraw(tools, ref, fix, carriers[fix.task_id].defect_status) and fix.status in _CANCELLABLE_FIX:
        tools.cancel(fix.task_id)


def converge_ruling(tools: ClosedTools, ref: str, record: RefRecord | None) -> _Outcome:
    """Carry Leo's portal ruling to the filer's own fix and gate: withdraw and cancel the fix first, since a cancelled gate releases it."""
    try:
        _withdraw_and_cancel_fix(tools, ref)
        for gate in _filer_tasks(tools, tools.find_gates(ref)):
            if gate.status not in _TERMINAL:
                tools.cancel(gate.task_id)
    except ToolRefused as refused:
        return _Outcome(refused=_unconverged(ref, refused, record), record=record)
    return _Outcome()


def _filer(policy: Policy) -> FilerPolicy:
    if policy.filer is None:
        raise ValueError(f'the policy of {policy.project_id} has no filer block')
    return policy.filer


def _read(policy: Policy, offset: int) -> tuple[InboxRead, int]:
    """The inbox past `offset`, and where the read began: a rewound inbox is read again from its start."""
    try:
        return read_inbox(policy.in_dir, offset=offset), offset
    except InboxUnusable as exc:
        if exc.problem is not FeedIoProblem.INBOX_REWOUND:
            raise
        logger.warning('inbox of %s was replaced (%s); reading it from its start', policy.project_id, exc.detail)
        return read_inbox(policy.in_dir, offset=0), 0


def run_pass(policy: Policy, tools: ClosedTools, cursor: FilerCursor) -> tuple[PassReport, FilerCursor]:
    """One pass over the inbox from the cursor; stateless, so the caller stores the cursor it returns."""
    try:
        read, offset_from = _read(policy, cursor.inbox_offset)
    except InboxUnusable as exc:
        unusable = PassReport(
            offset_from=cursor.inbox_offset,
            offset_to=cursor.inbox_offset,
            converged=(),
            unconverged=(),
            rejected_lines=0,
            inbox_problem=str(exc.problem),
        )
        return unusable, cursor
    outcomes: dict[str, _Outcome] = {}
    for ref, state in _ref_states(read, policy.project_id).items():
        record = cursor.refs.get(ref)
        if state.ruled or state.report is None:
            outcomes[ref] = converge_ruling(tools, ref, record)
        else:
            outcomes[ref] = converge_filing(tools, state.report, ref, policy, record)
    unconverged = tuple(outcome.refused for outcome in outcomes.values() if outcome.refused is not None)
    offset_to = offset_from if unconverged else read.offset
    report = PassReport(
        offset_from=offset_from,
        offset_to=offset_to,
        converged=tuple(ref for ref, outcome in outcomes.items() if outcome.refused is None),
        unconverged=unconverged,
        rejected_lines=len(read.rejected),
    )
    kept = {ref: outcome.record for ref, outcome in outcomes.items() if outcome.refused and outcome.record}
    return report, FilerCursor(inbox_offset=offset_to, refs=kept)


def _log_unconverged(report: PassReport) -> None:
    for item in report.unconverged:
        logger.warning(
            '%s not filed: %s refused (%s, %s), %s after %d attempts',
            item.ref,
            item.tool,
            item.kind,
            item.code,
            item.disposition,
            item.attempts,
        )


def file_once(policy: Policy, *, open_tools: OpenTools) -> PassReport:
    """One pass under the held state dir, from the stored cursor, storing the cursor it leaves."""
    with held_state_dir(policy) as state:
        cursor = state.read_cursor()
        with closed_tools(policy, open_tools) as tools:
            report, next_cursor = run_pass(policy, tools, cursor)
        state.write_cursor(next_cursor)
    _log_unconverged(report)
    return report


def next_delay(interval: int, transient_streak: int) -> float:
    """The wait before the next pass: the interval, doubled per consecutive transient pass up to MAX_BACKOFF_SECONDS."""
    cap = max(interval, MAX_BACKOFF_SECONDS)
    return float(min(interval * 2 ** min(transient_streak, cap.bit_length()), cap))


def _is_transient(report: PassReport) -> bool:
    return any(item.disposition is Disposition.RETRY for item in report.unconverged)


def _stored_pass(policy: Policy, state: StateDir, cursor: FilerCursor, open_tools: OpenTools) -> tuple[bool, FilerCursor]:
    """One pass over its own session, its cursor stored; whether it was transient, and the cursor it leaves."""
    try:
        with closed_tools(policy, open_tools) as tools:
            report, cursor = run_pass(policy, tools, cursor)
    except SessionUnavailable as exc:
        logger.warning('fused-memory of %s is unreachable (%s); the pass is retried', policy.project_id, exc.kind)
        transient = True
    else:
        _log_unconverged(report)
        transient = _is_transient(report)
    state.write_cursor(cursor)
    return transient, cursor


def run_until_stopped(
    policy: Policy,
    *,
    open_tools: OpenTools,
    stop: threading.Event,
    wait: Callable[[float], object] | None = None,
) -> None:
    """Pass after pass under the state dir held for the whole run, backing off while passes are transient, until `stop` is set."""
    pause = stop.wait if wait is None else wait
    interval = _filer(policy).interval_seconds
    streak = 0
    with held_state_dir(policy) as state:
        cursor = state.read_cursor()
        while not stop.is_set():
            transient, cursor = _stored_pass(policy, state, cursor, open_tools)
            streak = streak + 1 if transient else 0
            pause(next_delay(interval, streak))


class CheckedCall(BaseModel):
    """A call one pass would make: a read it performed, or a write it withheld."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    tool: str
    arguments: dict[str, Any]
    withheld: bool


class CheckReport(BaseModel):
    """σ4's install check: the closed tools the server lacks, the check ref's standing, and the calls one pass would make."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    ok: bool
    missing_tools: tuple[str, ...]
    check_ref: str
    check_ref_live: bool | None
    calls: tuple[CheckedCall, ...]
    problem: str | None


def _pass_problem(report: PassReport) -> str | None:
    if report.inbox_problem is not None:
        return f'the inbox is unusable ({report.inbox_problem})'
    refusals = [f'{item.ref}: {item.tool} refused ({item.kind}, {item.code})' for item in report.unconverged]
    return '; '.join(refusals) or None


def _check_report(
    ref: str,
    missing: tuple[str, ...],
    live: bool | None,
    withheld: WithheldWrites | None,
    problem: str | None,
) -> CheckReport:
    calls = () if withheld is None else withheld.calls
    return CheckReport(
        ok=not missing and problem is None,
        missing_tools=missing,
        check_ref=ref,
        check_ref_live=live,
        calls=tuple(CheckedCall(tool=call.tool, arguments=dict(call.arguments), withheld=call.withheld) for call in calls),
        problem=problem,
    )


def _checked(policy: Policy, session: ListingCaller, cursor: FilerCursor, ref: str) -> CheckReport:
    try:
        missing = missing_tools(session)
        live = bool(ClosedTools(session, policy.project_root).lookup_defect_ref(ref))
    except ToolRefused as refused:
        return _check_report(ref, (), None, None, str(refused))
    withheld = WithheldWrites(session)
    report, _ = run_pass(policy, ClosedTools(withheld, policy.project_root), cursor)
    return _check_report(ref, missing, live, withheld, _pass_problem(report))


def check(policy: Policy, *, open_session: OpenSession) -> CheckReport:
    """One pass's plan under the held state dir, its writes withheld and its cursor discarded: nothing is written but the lock file."""
    ref = check_ref(policy.project_id)
    with held_state_dir(policy) as state:
        cursor = state.read_cursor()
        try:
            with opened(policy, open_session) as session:
                return _checked(policy, session, cursor, ref)
        except SessionUnavailable as exc:
            return _check_report(ref, (), None, None, str(exc))
