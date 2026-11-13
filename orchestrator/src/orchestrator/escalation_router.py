"""The applier half of the info-L0 disposition router.

plans/info-l0-disposition-router-prd.md D5/D7/D8/D9/D12.  The decision half is
pure and lives in escalation/src/escalation/disposition.py::route_info_l0; this
module performs what a decision means: queue writes, curator tickets, the D9
hold a ticket leaves on its record, and the per-sweep aggregates.  Its three
callers (the workflow exit finalizer, the orphan reaper and the startup
amnesty) only wire it.  A fourth consumer, the reviewer leg (D2), is applied
here too: the reviewer's verdict dispositions its own task's notes mid-run,
at no exit, through ``apply_reviewer_disposition``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Protocol, TypeVar, assert_never

from escalation.amendments import amendment_room
from escalation.categories import EscalationCategory
from escalation.classify import INFO_L0_MECHANICAL_ROLES, info_l0_mechanical_class, is_harness_filed
from escalation.disposition import (
    Addressed,
    ConvertViaCurator,
    CuratorCandidate,
    Defer,
    Disposition,
    ExitKind,
    ObservationConsumed,
    PromoteReason,
    PromoteWithHint,
    ReviewerDisposition,
    StatusInfo,
    route_info_l0,
)
from escalation.models import Escalation, normalised_severity
from escalation.queue import EscalationQueue, ResolveOutcome
from shared.eval_lane import eval_lane_provenance
from shared.mcp_envelope import parse_tool_result
from shared.task_submit import (
    SubmitFailed,
    SubmittedTask,
    SubmittedTicket,
    parse_submit_task_result,
)

from orchestrator.event_store import EventStore, EventType
from orchestrator.review_payload import NoteDisposition
from orchestrator.workflow_types import WorkflowOutcome

logger = logging.getLogger(__name__)

ROUTER_ID = 'info-l0-router'
REVIEWER_LEG_ID = 'reviewer-comprehensive/workflow'
AGGREGATE_ROLE = 'harness-info-l0-router'
FALLBACK_ACTION = 'convert_to_task'
AGGREGATE_CATEGORY = 'risk_identified'
CONVERTED_CLASS = 'converted'


def router_owns(record: Escalation, *, enabled: bool) -> bool:
    """True when the router, not a steward or the eval harness, is *record*'s exit consumer."""
    return (
        enabled
        and record.level == 0
        and normalised_severity(record.severity) == 'info'
        and eval_lane_provenance(record.task_id, record.worktree) is None
    )


_EXIT_KIND_BY_OUTCOME: dict[WorkflowOutcome, ExitKind] = {
    WorkflowOutcome.DONE: ExitKind.DONE,
    WorkflowOutcome.MERGE_DEFERRED: ExitKind.MERGE_DEFERRED,
    WorkflowOutcome.BLOCKED: ExitKind.BLOCKED,
    WorkflowOutcome.ESCALATED: ExitKind.BLOCKED,
    WorkflowOutcome.CANCELLED: ExitKind.CANCELLED,
    WorkflowOutcome.SOFT_CANCELLED: ExitKind.CANCELLED,
    WorkflowOutcome.REQUEUED: ExitKind.REQUEUED,
    WorkflowOutcome.PLANNED: ExitKind.REQUEUED,
}


def exit_kind_for(
    cancel_kind: Literal['hard', 'soft'] | None, outcome: WorkflowOutcome | None,
) -> ExitKind | None:
    """The exit a workflow made, or None when it is unknown and nothing may be routed."""
    if cancel_kind is not None:
        return ExitKind.CANCELLED
    if outcome is None:
        return None
    return _EXIT_KIND_BY_OUTCOME[outcome]


_Stamped = TypeVar('_Stamped')


def _stamp_note(prefix: str, fields: Mapping[str, str]) -> str:
    return prefix + json.dumps(dict(fields))


def _read_stamp(
    record: Escalation, *, owner: str, prefix: str, kind: str,
    parse: Callable[[Mapping[str, object]], _Stamped],
) -> _Stamped | None:
    """What *owner* stamped into *record*'s triage note under *prefix*, or None.

    The one parser of ``_stamp_note``: a note *owner* stamped that does not
    parse logs a WARNING and reads as unstamped.
    """
    if record.triaged_by != owner:
        return None
    note = record.triage_note or ''
    try:
        if not note.startswith(prefix):
            raise ValueError(f'note does not carry the {kind} prefix')
        fields = json.loads(note.removeprefix(prefix))
        if not isinstance(fields, dict):
            raise ValueError(f'{kind} fields are not an object')
        return parse(fields)
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning(
            'info-l0-router: %s is stamped by %s but its triage note is not a %s (%s); '
            'it is routed as if unstamped. note=%r',
            record.id, owner, kind, exc, note[:200],
        )
        return None


def _string_field(fields: Mapping[str, object], name: str) -> str:
    value = fields[name]
    if not isinstance(value, str):
        raise ValueError(f'{name} is not a string')
    return value


@dataclass(frozen=True)
class RouterHold:
    """A curator ticket filed for a record that stays pending until the ticket resolves (D9)."""

    ticket: str
    filed_at: datetime


_HOLD_PREFIX = 'info-l0-router hold: '


def hold_note(ticket: str, filed_at: datetime) -> str:
    return _stamp_note(_HOLD_PREFIX, {'ticket': ticket, 'filed_at': filed_at.isoformat()})


def read_hold(record: Escalation) -> RouterHold | None:
    """The router's hold on *record*, or None; the only reader of ``hold_note``."""
    return _read_stamp(
        record, owner=ROUTER_ID, prefix=_HOLD_PREFIX, kind='router hold', parse=_hold_from,
    )


def _hold_from(fields: Mapping[str, object]) -> RouterHold:
    filed = datetime.fromisoformat(_string_field(fields, 'filed_at'))
    if filed.tzinfo is None:
        raise ValueError('hold filed_at carries no timezone')
    return RouterHold(_string_field(fields, 'ticket'), filed)


@dataclass(frozen=True)
class ReviewerMark:
    """The reviewer's standing verdict on a note it left pending, and its reason (D2)."""

    disposition: ReviewerDisposition
    rationale: str


_REVIEWER_MARK_PREFIX = 'reviewer verdict: '


def reviewer_mark_note(mark: ReviewerMark) -> str:
    return _stamp_note(
        _REVIEWER_MARK_PREFIX, {'disposition': str(mark.disposition), 'rationale': mark.rationale},
    )


def read_reviewer_mark(record: Escalation) -> ReviewerMark | None:
    """The reviewer's mark on *record*, or None; the only reader of ``reviewer_mark_note``."""
    return _read_stamp(
        record, owner=REVIEWER_LEG_ID, prefix=_REVIEWER_MARK_PREFIX, kind='reviewer mark',
        parse=_reviewer_mark_from,
    )


def _reviewer_mark_from(fields: Mapping[str, object]) -> ReviewerMark:
    return ReviewerMark(
        ReviewerDisposition(_string_field(fields, 'disposition')),
        _string_field(fields, 'rationale'),
    )


def _carrying_rationale(decided: Disposition, mark: ReviewerMark | None) -> Disposition:
    """*decided*, with the reviewer's reason for a marked note handed to the curator too."""
    if mark is None or not isinstance(decided, ConvertViaCurator):
        return decided
    candidate = decided.payload
    return ConvertViaCurator(replace(
        candidate,
        description=f'{candidate.description}\n\nReviewer rationale ({REVIEWER_LEG_ID}): {mark.rationale}',
    ))


@dataclass(frozen=True)
class Converted:
    task_id: str


@dataclass(frozen=True)
class Refused:
    reason: str


@dataclass(frozen=True)
class Failed:
    reason: str


@dataclass(frozen=True)
class Undecided:
    reason: str


TicketVerdict = Converted | Refused | Failed | Undecided

# fused-memory answers these from a short wait or a closing store without
# changing the ticket row, so the curator may still decide it.
_UNDECIDED_FAILURE_REASONS = frozenset({'timeout', 'server_closed'})


def classify_ticket(rpc_body: object) -> TicketVerdict:
    """Classify a ``resolve_ticket`` JSON-RPC body.  Never raises."""
    answer, err = parse_tool_result(rpc_body, None, dict)
    if err is not None:
        return Undecided(f'unreadable resolve_ticket answer ({err.shape})')
    assert answer is not None  # parse_tool_result invariant: (None, err) | (value, None)
    status = answer.get('status')
    reason = str(answer.get('reason') or '')
    task_id = answer.get('task_id')
    if status in ('created', 'combined') and isinstance(task_id, str | int) and str(task_id):
        return Converted(str(task_id))
    if status == 'refused':
        return Refused(reason)
    if status == 'failed' and reason not in _UNDECIDED_FAILURE_REASONS:
        return Failed(reason)
    return Undecided(f'status={status!r} reason={reason!r}')


def status_info_sentinel(class_key: str) -> str:
    """The task id that keys one status-info class's aggregate L1 (D8)."""
    return f'__info_l0_status_info_{class_key}__'


OVERFLOW_SENTINEL = '__info_l0_conversion_overflow__'


class DispatchTool(Protocol):
    """The shape of ``orchestrator/src/orchestrator/scheduler.py::Scheduler.dispatch_tool``."""

    async def __call__(self, name: str, arguments: dict, *, timeout: float = ...) -> dict: ...


@dataclass(frozen=True)
class RouterPorts:
    """Everything one routing decision may touch, bundled once per caller."""

    queue: EscalationQueue
    event_store: EventStore | None
    dispatch_tool: DispatchTool
    project_root: str
    filing_claimant_run_id: str | None


@dataclass(frozen=True)
class RouteContext:
    by: str
    exit_kind: ExitKind


class Outcome(StrEnum):
    """What ``apply`` or ``apply_reviewer_disposition`` did to one record."""

    CLOSED = 'closed'
    ALREADY_TERMINAL = 'already-terminal'
    CONVERT_PENDING = 'convert-pending'
    PROMOTED = 'promoted'
    DEFERRED = 'defer'
    ROUTER_ERROR = 'router-error'
    REJECTED = 'rejected_disposition'


DISPATCH_TIMEOUT_SECS = 15.0
RESOLVE_WAIT_SECS = 2.0


@dataclass(frozen=True)
class Applied:
    """What ``apply`` did to one record; a close also names the class it stamped."""

    escalation_id: str
    outcome: Outcome
    resolution_class: str | None = None
    reason: str | None = None
    ticket: str | None = None
    promoted_to: str | None = None
    curator_unreachable: bool = False

    def __post_init__(self) -> None:
        if (self.outcome is Outcome.CLOSED) != (self.resolution_class is not None):
            raise ValueError(
                f'Applied for {self.escalation_id}: resolution_class must be set exactly '
                f'when the outcome is closed (outcome={self.outcome}, '
                f'resolution_class={self.resolution_class!r})'
            )

    @property
    def event_class(self) -> str:
        """The ``info_l0_dispositioned`` event's ``class``: the stamped class, else the outcome."""
        return self.resolution_class or str(self.outcome)


@dataclass(frozen=True)
class _HoldOutcome:
    """The curator's answer to a D9 hold, performed like a disposition."""

    hold: RouterHold
    verdict: TicketVerdict


_Action = Disposition | _HoldOutcome


async def apply(
    disposition: Disposition, record: Escalation, ports: RouterPorts, context: RouteContext,
) -> Applied:
    """Perform *disposition* on *record*.  Never raises (INV-11).

    The record is re-read first, and nothing is written for one that is no
    longer pending (INV-3).  A failure becomes a ``router_error`` promotion.
    """
    return await _apply(disposition, record, ports, context)


async def _apply(
    action: _Action, record: Escalation, ports: RouterPorts, context: RouteContext,
) -> Applied:
    try:
        current = ports.queue.get(record.id)
        if current is None or current.status != 'pending':
            return Applied(record.id, Outcome.ALREADY_TERMINAL)
        applied = await _perform(action, current, ports, context)
    except Exception:
        logger.exception(
            'info-l0-router: applying %r to %s failed (by=%s, exit_kind=%s)',
            action, record.id, context.by, context.exit_kind,
            extra={
                'escalation_id': record.id, 'disposition': repr(action),
                'by': context.by, 'exit_kind': str(context.exit_kind),
            },
        )
        applied = _recover(action, record, ports, context)
    _emit(applied, record.task_id, ports, by=context.by, exit_kind=context.exit_kind)
    return applied


async def _perform(
    action: _Action, record: Escalation, ports: RouterPorts, context: RouteContext,
) -> Applied:
    match action:
        case Addressed() | ObservationConsumed():
            return _close(record, action.resolution_class, '', ports, context)
        case StatusInfo(class_key=class_key):
            note = (
                f': mechanical {class_key} notice, listed in the aggregate for '
                f'{status_info_sentinel(class_key)}'
            )
            closed = _close(record, StatusInfo.resolution_class, note, ports, context)
            return replace(closed, reason=class_key)
        case ConvertViaCurator(payload=candidate):
            return await _convert(record, candidate, ports, context)
        case PromoteWithHint(reason=reason):
            return _promote(record, reason, ports, context)
        case Defer(reason=reason):
            return Applied(record.id, Outcome.DEFERRED, reason=str(reason))
        case _HoldOutcome(hold=hold, verdict=verdict):
            return _finish(record, hold, verdict, ports, context)
        case _:
            assert_never(action)


def _close(
    record: Escalation, resolution_class: str, note: str, ports: RouterPorts,
    context: RouteContext,
) -> Applied:
    return _write_close(
        ports, record.id, resolution_class,
        resolution=(
            f'{resolution_class} — closed by the info-L0 router ({context.by}, exit '
            f'{context.exit_kind}){note}'
        ),
        resolved_by=ROUTER_ID,
    )


def _write_close(
    ports: RouterPorts, record_id: str, resolution_class: str, *, resolution: str,
    resolved_by: str,
) -> Applied:
    """Dismiss *record_id*; ALREADY_TERMINAL when another writer closed it first."""
    closure = _resolve_outcome()
    ports.queue.resolve(
        record_id, resolution,
        dismiss=True, resolved_by=resolved_by, resolution_class=resolution_class,
        outcome=closure,
    )
    if not closure['applied']:
        return Applied(record_id, Outcome.ALREADY_TERMINAL)
    return Applied(record_id, Outcome.CLOSED, resolution_class=resolution_class)


def _resolve_outcome() -> ResolveOutcome:
    return {
        'applied': False, 'prior_status': None, 'prior_resolved_by': None,
        'late_resolution_captured': False, 'resolution_class_corrected': None,
    }


async def _convert(
    record: Escalation, candidate: CuratorCandidate, ports: RouterPorts, context: RouteContext,
) -> Applied:
    try:
        response = await ports.dispatch_tool(
            'submit_task', _submit_arguments(candidate, ports.project_root),
            timeout=DISPATCH_TIMEOUT_SECS,
        )
    except Exception as exc:
        logger.warning('info-l0-router: submit_task for %s raised', record.id, exc_info=True)
        cause = f'submit_task dispatch raised {type(exc).__name__}: {exc}'
        applied = _promote(record, PromoteReason.TICKET_FAILED, ports, context, cause=cause)
        return replace(applied, curator_unreachable=True)
    match parse_submit_task_result(response):
        case SubmittedTicket(ticket_id=ticket):
            held = ports.queue.stamp_triage(
                record.id, triaged_by=ROUTER_ID, triage_note=hold_note(ticket, datetime.now(UTC)),
            )
            if held is None:
                return Applied(record.id, Outcome.ALREADY_TERMINAL, ticket=ticket)
            return Applied(record.id, Outcome.CONVERT_PENDING, ticket=ticket)
        case SubmittedTask(task_id=task_id):
            return _close_converted(record, task_id, None, ports, context)
        case SubmitFailed(reason=cause):
            return _promote(record, PromoteReason.TICKET_FAILED, ports, context, cause=cause)
        case unreachable:
            assert_never(unreachable)


def _submit_arguments(candidate: CuratorCandidate, project_root: str) -> dict:
    return {
        'project_root': project_root,
        'title': candidate.title,
        'description': candidate.description,
        'priority': 'low',
        'planning_mode': False,
        'metadata': {
            'source': ROUTER_ID,
            'spawn_context': ROUTER_ID,
            'escalation_id': candidate.escalation_id,
            'spawned_from': candidate.spawned_from,
            'suggestion_hash': _suggestion_hash(candidate.escalation_id),
        },
    }


def _suggestion_hash(escalation_id: str) -> str:
    return hashlib.sha256(f'{ROUTER_ID}:{escalation_id}'.encode()).hexdigest()[:16]


def _close_converted(
    record: Escalation, task_id: str, ticket: str | None, ports: RouterPorts,
    context: RouteContext,
) -> Applied:
    via = f' via curator ticket {ticket}' if ticket else ''
    closed = _close(record, CONVERTED_CLASS, f': converted:{task_id}{via}', ports, context)
    return replace(closed, ticket=ticket)


def _finish(
    record: Escalation, hold: RouterHold, verdict: TicketVerdict, ports: RouterPorts,
    context: RouteContext,
) -> Applied:
    match verdict:
        case Converted(task_id=task_id):
            return _close_converted(record, task_id, hold.ticket, ports, context)
        case Refused(reason=why):
            reason = PromoteReason.TICKET_REFUSED
        case Failed(reason=why):
            reason = PromoteReason.TICKET_FAILED
        case Undecided(reason=why):
            reason = PromoteReason.TICKET_TIMEOUT
        case _:
            assert_never(verdict)
    cause = f'curator ticket {hold.ticket}: {why}'
    return replace(_promote(record, reason, ports, context, cause=cause), ticket=hold.ticket)


def _promote(
    record: Escalation, reason: PromoteReason, ports: RouterPorts, context: RouteContext,
    *, cause: str = '',
) -> Applied:
    """The D5 fallback: the L0 dismissed citing one L1 that hints conversion.

    The dismissal is written first, so a record no pass can dismiss is never
    promoted again by the next one; an L1 that then fails to file leaves the
    note's full text in the ERROR log.
    """
    l1 = Escalation(
        id=ports.queue.make_id(record.task_id),
        task_id=record.task_id,
        agent_role=AGGREGATE_ROLE,
        severity=record.severity,
        category=_filing_category(record.category),
        summary=f'Info note not converted to a task ({reason}): {record.summary}'[:200],
        detail=_fallback_detail(record, reason, cause, context),
        suggested_action=FALLBACK_ACTION,
        workflow_state=record.workflow_state,
        level=1,
        filing_claimant_run_id=ports.filing_claimant_run_id,
    )
    dismissal = _resolve_outcome()
    ports.queue.resolve(
        record.id,
        f'Promoted by the info-L0 router ({context.by}) to {l1.id}: {reason}',
        dismiss=True, resolved_by=ROUTER_ID, outcome=dismissal,
    )
    if not dismissal['applied']:
        return Applied(record.id, Outcome.ALREADY_TERMINAL, reason=str(reason))
    try:
        ports.queue.submit(l1)
    except Exception:
        logger.exception(
            'info-l0-router: dismissed %s but could not file its fallback L1 %s; the '
            'note survives only in this log line: %s',
            record.id, l1.id, l1.detail,
        )
        return Applied(record.id, Outcome.ROUTER_ERROR, reason=str(reason))
    return Applied(record.id, Outcome.PROMOTED, reason=str(reason), promoted_to=l1.id)


def _filing_category(category: str) -> str:
    try:
        return EscalationCategory(category).value
    except ValueError:
        return AGGREGATE_CATEGORY


def _fallback_detail(
    record: Escalation, reason: PromoteReason, cause: str, context: RouteContext,
) -> str:
    why = f'{reason} ({cause})' if cause else str(reason)
    lines = [
        record.detail or record.summary,
        '',
        f'[info-l0-router] This info note was not converted to a task: {why}.',
        f'Source escalation: {record.id}, filed by {record.agent_role}, routed by '
        f'{context.by} at exit {context.exit_kind}.',
        f"Author's suggested_action: {record.suggested_action or '(none)'}",
    ]
    if _filing_category(record.category) != record.category:
        lines.append(f'Original category: {record.category!r} (unregistered).')
    lines.append(
        f'Convert it into a task (submit_task with metadata.escalation_id={record.id}) '
        'or close it with a rationale.'
    )
    return '\n'.join(lines)


def _recover(
    action: _Action, record: Escalation, ports: RouterPorts, context: RouteContext,
) -> Applied:
    if isinstance(action, PromoteWithHint):
        return Applied(record.id, Outcome.ROUTER_ERROR, reason=str(action.reason))
    try:
        return _promote(
            record, PromoteReason.ROUTER_ERROR, ports, context,
            cause=f'applying {action!r} failed',
        )
    except Exception:
        logger.exception(
            'info-l0-router: could not file the router_error L1 for %s (by=%s)',
            record.id, context.by,
        )
        return Applied(record.id, Outcome.ROUTER_ERROR, reason=str(PromoteReason.ROUTER_ERROR))


def _emit(
    applied: Applied, task_id: str, ports: RouterPorts, *, by: str, exit_kind: ExitKind | None,
) -> None:
    if ports.event_store is None:
        return
    try:
        ports.event_store.emit(
            EventType.info_l0_dispositioned,
            task_id=task_id,
            data={
                'escalation_id': applied.escalation_id,
                'class': applied.event_class,
                'by': by,
                'exit_kind': None if exit_kind is None else str(exit_kind),
                'ticket': applied.ticket,
                'task_id': task_id,
                'decided_at': datetime.now(UTC).isoformat(),
                'reason': applied.reason,
                'promoted_to': applied.promoted_to,
            },
        )
    except Exception:
        logger.warning(
            'info-l0-router: event emit failed for %s', applied.escalation_id, exc_info=True,
        )


def reviewer_note_refusal(record: Escalation | None, *, task_id: str, enabled: bool) -> str | None:
    """Why the reviewer of *task_id* may not see or disposition *record*, or None when it may.

    The one eligibility rule for the reviewer leg (D2): the briefing lists
    exactly the notes it admits, and a disposition of any other is rejected.
    D2 admits agent-filed notes only, so the harness namespace
    (escalation/classify.py::is_harness_filed) is refused.
    """
    if record is None:
        return 'no such escalation'
    if record.task_id != task_id:
        return f'filed for task {record.task_id!r}, not the reviewed task {task_id!r}'
    if record.status != 'pending':
        return f'status is {record.status!r}, not pending'
    if not router_owns(record, enabled=enabled):
        return (
            f'not an info L0 the router owns (severity {record.severity!r}, level '
            f'{record.level}, router {"enabled" if enabled else "disabled"}, eval lane '
            f'{eval_lane_provenance(record.task_id, record.worktree)!r})'
        )
    mechanical_class = info_l0_mechanical_class(record)
    if mechanical_class is not None:
        return f'a mechanical {mechanical_class!r} notice, which the exit router closes'
    if is_harness_filed(record):
        return f'filed by the harness under role {record.agent_role!r}, not by an agent'
    hold = read_hold(record)
    if hold is not None:
        return f'held by the router on curator ticket {hold.ticket}'
    return None


def reviewer_notes(queue: EscalationQueue, *, task_id: str, enabled: bool) -> list[Escalation]:
    """The open info notes the reviewer of *task_id* is shown and may disposition."""
    return [
        record for record in queue.get_by_task(task_id, status='pending', level=0)
        if reviewer_note_refusal(record, task_id=task_id, enabled=enabled) is None
    ]


def apply_reviewer_disposition(
    note: NoteDisposition, ports: RouterPorts, *, task_id: str, enabled: bool,
) -> Applied:
    """Perform the reviewer's verdict on one of its task's notes (D2).  Never raises.

    A close dismisses the note; ``work`` leaves it pending and stamps the
    reviewer mark (``read_reviewer_mark``) that the exit router honours; every
    other path writes nothing, and the exit router remains the backstop.
    """
    try:
        applied = _apply_reviewer(note, ports, task_id=task_id, enabled=enabled)
    except Exception as exc:
        logger.exception(
            'info-l0-router: applying the reviewer disposition %s to %s failed (task %s); '
            'it stays pending for the exit router',
            note.disposition, note.escalation_id, task_id,
            extra={
                'escalation_id': note.escalation_id, 'disposition': str(note.disposition),
                'by': REVIEWER_LEG_ID,
            },
        )
        applied = Applied(
            note.escalation_id, Outcome.ROUTER_ERROR, reason=f'{type(exc).__name__}: {exc}',
        )
    _emit(applied, task_id, ports, by=REVIEWER_LEG_ID, exit_kind=None)
    return applied


def _apply_reviewer(
    note: NoteDisposition, ports: RouterPorts, *, task_id: str, enabled: bool,
) -> Applied:
    refusal = reviewer_note_refusal(
        ports.queue.get(note.escalation_id), task_id=task_id, enabled=enabled,
    )
    if refusal is not None:
        logger.warning(
            'info-l0-router: rejected the reviewer disposition %s of %s (task %s): %s',
            note.disposition, note.escalation_id, task_id, refusal,
            extra={
                'escalation_id': note.escalation_id, 'disposition': str(note.disposition),
                'by': REVIEWER_LEG_ID, 'reason': refusal,
            },
        )
        return Applied(note.escalation_id, Outcome.REJECTED, reason=refusal)
    match note.disposition:
        case ReviewerDisposition.ADDRESSED:
            resolution_class = Addressed.resolution_class
        case ReviewerDisposition.NO_ACTION:
            resolution_class = ObservationConsumed.resolution_class
        case ReviewerDisposition.WORK:
            marked = ports.queue.stamp_triage(
                note.escalation_id, triaged_by=REVIEWER_LEG_ID,
                triage_note=reviewer_mark_note(ReviewerMark(note.disposition, note.rationale)),
            )
            if marked is None:
                return Applied(note.escalation_id, Outcome.ALREADY_TERMINAL)
            return Applied(note.escalation_id, Outcome.DEFERRED, reason=str(note.disposition))
        case _:
            assert_never(note.disposition)
    return _write_close(
        ports, note.escalation_id, resolution_class,
        resolution=f'{resolution_class} — dispositioned by the reviewer: {note.rationale}',
        resolved_by=REVIEWER_LEG_ID,
    )


class InfoL0Sweep:
    """One routing pass over info L0s, for one caller.

    The state is the pass's own: the D12 conversion cap, the curator
    short-circuit (after one unreachable dispatch no further dispatch is
    made), and the records ``close`` lists in the D8/D12 aggregates.
    """

    def __init__(self, ports: RouterPorts, *, by: str, max_conversions: int) -> None:
        self._ports = ports
        self._by = by
        self._max_conversions = max_conversions
        self._conversions = 0
        self._curator_unreachable = False
        self._status_info: dict[str, list[Escalation]] = {}
        self._overflow: list[Escalation] = []
        self._tally: Counter[str] = Counter()

    async def route(
        self, record: Escalation, *, exit_kind: ExitKind, task_status: str | None = None,
    ) -> Applied | None:
        """Decide and apply *record*'s disposition; None when it is held or no longer pending."""
        current = self._ports.queue.get(record.id)
        if current is None or current.status != 'pending' or read_hold(current) is not None:
            return None
        mark = read_reviewer_mark(current)
        try:
            decided = _carrying_rationale(route_info_l0(
                current, task_status=task_status, exit_kind=exit_kind,
                mechanical_roles=INFO_L0_MECHANICAL_ROLES,
                reviewer_disposition=None if mark is None else mark.disposition,
            ), mark)
        except Exception:
            logger.exception('info-l0-router: deciding %s failed (by=%s)', current.id, self._by)
            decided = PromoteWithHint(PromoteReason.ROUTER_ERROR)
        disposition = self._admit(current, decided)
        applied = await apply(disposition, current, self._ports, RouteContext(self._by, exit_kind))
        self._note(current, disposition, applied)
        return applied

    async def finish_hold(
        self, record: Escalation, hold: RouterHold, *, now: datetime, timeout_secs: float,
        exit_kind: ExitKind = ExitKind.ORPHAN,
    ) -> Applied | None:
        """Close a D9 hold the curator decided, or one past its bound; None while it may still decide."""
        verdict = await self._ask_curator(hold)
        if isinstance(verdict, Undecided) and (now - hold.filed_at).total_seconds() < timeout_secs:
            return None
        applied = await _apply(
            _HoldOutcome(hold, verdict), record, self._ports, RouteContext(self._by, exit_kind),
        )
        self._tally[applied.event_class] += 1
        return applied

    def close(self) -> None:
        """File or amend this pass's aggregates and log its tally.  Never raises."""
        for class_key, records in self._status_info.items():
            self._file_or_amend(
                status_info_sentinel(class_key), records,
                summary=f'{len(records)} {class_key} info notice(s) closed status-info ({self._by})',
                suggested_action='acknowledge_status_info',
            )
        if self._overflow:
            self._file_or_amend(
                OVERFLOW_SENTINEL, self._overflow,
                summary=(
                    f'{len(self._overflow)} info note(s) over the per-sweep conversion cap of '
                    f'{self._max_conversions} promoted with {FALLBACK_ACTION} ({self._by})'
                ),
                suggested_action=FALLBACK_ACTION,
            )
        if self._tally:
            logger.info('info-l0-router (%s): %s', self._by, dict(self._tally))
        self._status_info, self._overflow, self._tally = {}, [], Counter()

    def _admit(self, record: Escalation, disposition: Disposition) -> Disposition:
        if not isinstance(disposition, ConvertViaCurator):
            return disposition
        if self._curator_unreachable:
            return PromoteWithHint(PromoteReason.TICKET_FAILED)
        if self._conversions >= self._max_conversions:
            self._overflow.append(record)
            return PromoteWithHint(PromoteReason.CONVERSION_CAP)
        self._conversions += 1
        return disposition

    def _note(self, record: Escalation, disposition: Disposition, applied: Applied) -> None:
        self._tally[applied.event_class] += 1
        if applied.curator_unreachable:
            self._curator_unreachable = True
        if isinstance(disposition, StatusInfo) and applied.outcome is Outcome.CLOSED:
            self._status_info.setdefault(disposition.class_key, []).append(record)

    async def _ask_curator(self, hold: RouterHold) -> TicketVerdict:
        if self._curator_unreachable:
            return Undecided('the curator was unreachable earlier in this sweep')
        try:
            body = await self._ports.dispatch_tool(
                'resolve_ticket',
                {
                    'ticket': hold.ticket, 'project_root': self._ports.project_root,
                    'timeout_seconds': RESOLVE_WAIT_SECS,
                },
                timeout=DISPATCH_TIMEOUT_SECS,
            )
        except Exception as exc:
            logger.warning('info-l0-router: resolve_ticket %s raised', hold.ticket, exc_info=True)
            self._curator_unreachable = True
            return Undecided(f'resolve_ticket dispatch raised {type(exc).__name__}')
        return classify_ticket(body)

    def _file_or_amend(
        self, sentinel: str, records: list[Escalation], *, summary: str, suggested_action: str,
    ) -> None:
        """Amend the newest open aggregate for *sentinel*, or file a fresh one.

        A fresh one is filed when none is open, when the newest has no
        amendment left before the queue would shed an earlier pass's listing,
        and when the amend did not land (the aggregate closed meanwhile).
        """
        queue = self._ports.queue
        listing = '\n'.join(f'{r.id} (task {r.task_id})' for r in records)
        try:
            newest = max(
                (
                    esc for esc in queue.get_by_task(
                        sentinel, status='pending', level=1, agent_role=AGGREGATE_ROLE,
                    )
                    if esc.category == AGGREGATE_CATEGORY
                ),
                key=lambda esc: esc.timestamp, default=None,
            )
            if newest is not None and amendment_room(newest) > 0:
                amended = queue.amend(
                    newest.id, agent_role=AGGREGATE_ROLE, summary=summary, detail=listing,
                )
                if amended['status'] in ('amended', 'repeat_framing'):
                    return
            queue.submit(Escalation(
                id=queue.make_id(sentinel),
                task_id=sentinel,
                agent_role=AGGREGATE_ROLE,
                severity='info',
                category=AGGREGATE_CATEGORY,
                summary=summary[:200],
                detail=f'{summary}.  Records:\n{listing}',
                suggested_action=suggested_action,
                level=1,
                filing_claimant_run_id=self._ports.filing_claimant_run_id,
            ))
        except Exception:
            logger.exception('info-l0-router: filing or amending the aggregate for %s failed', sentinel)
