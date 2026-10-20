"""The applier half of the info-L0 disposition router.

plans/info-l0-disposition-router-prd.md D5/D7/D8/D9/D12.  The decision half is
pure and lives in escalation/src/escalation/disposition.py::route_info_l0; this
module performs what a decision means: queue writes, curator tickets, the D9
hold a ticket leaves on its record, and the per-sweep aggregates.  Its three
callers (the workflow exit finalizer, the orphan reaper and the startup
amnesty) only wire it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal, Protocol, assert_never

from escalation.categories import EscalationCategory
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
    StatusInfo,
)
from escalation.models import Escalation, normalised_severity
from escalation.queue import EscalationQueue
from shared.mcp_envelope import parse_tool_result
from shared.task_submit import (
    SubmitFailed,
    SubmittedTask,
    SubmittedTicket,
    parse_submit_task_result,
)

from orchestrator.event_store import EventStore, EventType
from orchestrator.workflow_types import WorkflowOutcome

logger = logging.getLogger(__name__)

ROUTER_ID = 'info-l0-router'
AGGREGATE_ROLE = 'harness-info-l0-router'
FALLBACK_ACTION = 'convert_to_task'
AGGREGATE_CATEGORY = 'risk_identified'
CONVERTED_CLASS = 'converted'


def router_owns(record: Escalation, *, enabled: bool) -> bool:
    """True when the router, not a steward, is *record*'s exit consumer."""
    return enabled and record.level == 0 and normalised_severity(record.severity) == 'info'


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


@dataclass(frozen=True)
class RouterHold:
    """A curator ticket filed for a record that stays pending until the ticket resolves (D9)."""

    ticket: str
    filed_at: datetime


_HOLD_PREFIX = 'info-l0-router hold: '


def hold_note(ticket: str, filed_at: datetime) -> str:
    return _HOLD_PREFIX + json.dumps({'ticket': ticket, 'filed_at': filed_at.isoformat()})


def read_hold(record: Escalation) -> RouterHold | None:
    """The router's hold on *record*, or None; the only parser of ``hold_note``."""
    if record.triaged_by != ROUTER_ID:
        return None
    note = record.triage_note or ''
    try:
        if not note.startswith(_HOLD_PREFIX):
            raise ValueError('note does not carry the hold prefix')
        fields = json.loads(note.removeprefix(_HOLD_PREFIX))
        ticket, filed_at = fields['ticket'], fields['filed_at']
        if not isinstance(ticket, str) or not isinstance(filed_at, str):
            raise ValueError('hold fields are not strings')
        return RouterHold(ticket, datetime.fromisoformat(filed_at))
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning(
            'info-l0-router: %s is stamped by the router but its triage note is not a '
            'router hold (%s); it will be routed afresh. note=%r',
            record.id, exc, note[:200],
        )
        return None


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


ALREADY_TERMINAL = 'already-terminal'
CONVERT_PENDING = 'convert-pending'
PROMOTED = 'promoted'
DEFERRED = 'defer'
ROUTER_ERROR = 'router-error'
SUBMIT_TIMEOUT_SECS = 15.0


@dataclass(frozen=True)
class Applied:
    """What ``apply`` did to one record; ``disposition_class`` is the event's ``class``."""

    escalation_id: str
    disposition_class: str
    reason: str | None = None
    ticket: str | None = None
    promoted_to: str | None = None
    curator_unreachable: bool = False


async def apply(
    disposition: Disposition, record: Escalation, ports: RouterPorts, context: RouteContext,
) -> Applied:
    """Perform *disposition* on *record*.  Never raises (INV-11).

    The record is re-read first, and nothing is written for one that is no
    longer pending (INV-3).  A failure becomes a ``router_error`` promotion.
    """
    try:
        current = ports.queue.get(record.id)
        if current is None or current.status != 'pending':
            return Applied(record.id, ALREADY_TERMINAL)
        applied = await _perform(disposition, current, ports, context)
    except Exception:
        logger.exception(
            'info-l0-router: applying %r to %s failed (by=%s, exit_kind=%s)',
            disposition, record.id, context.by, context.exit_kind,
            extra={
                'escalation_id': record.id, 'disposition': repr(disposition),
                'by': context.by, 'exit_kind': str(context.exit_kind),
            },
        )
        applied = _recover(disposition, record, ports, context)
    _emit(applied, record, ports, context)
    return applied


async def _perform(
    disposition: Disposition, record: Escalation, ports: RouterPorts, context: RouteContext,
) -> Applied:
    match disposition:
        case Addressed() | ObservationConsumed():
            _close(record, disposition.resolution_class, '', ports, context)
            return Applied(record.id, disposition.resolution_class)
        case StatusInfo(class_key=class_key):
            note = (
                f': mechanical {class_key} notice, listed in the aggregate for '
                f'{status_info_sentinel(class_key)}'
            )
            _close(record, StatusInfo.resolution_class, note, ports, context)
            return Applied(record.id, StatusInfo.resolution_class, reason=class_key)
        case ConvertViaCurator(payload=candidate):
            return await _convert(record, candidate, ports, context)
        case PromoteWithHint(reason=reason):
            return _promote(record, reason, ports, context)
        case Defer(reason=reason):
            return Applied(record.id, DEFERRED, reason=str(reason))
        case _:
            assert_never(disposition)


def _close(
    record: Escalation, resolution_class: str, note: str, ports: RouterPorts,
    context: RouteContext,
) -> None:
    ports.queue.resolve(
        record.id,
        f'{resolution_class} — closed by the info-L0 router ({context.by}, exit '
        f'{context.exit_kind}){note}',
        dismiss=True, resolved_by=ROUTER_ID, resolution_class=resolution_class,
    )


async def _convert(
    record: Escalation, candidate: CuratorCandidate, ports: RouterPorts, context: RouteContext,
) -> Applied:
    try:
        response = await ports.dispatch_tool(
            'submit_task', _submit_arguments(candidate, ports.project_root),
            timeout=SUBMIT_TIMEOUT_SECS,
        )
    except Exception as exc:
        logger.warning('info-l0-router: submit_task for %s raised', record.id, exc_info=True)
        cause = f'submit_task dispatch raised {type(exc).__name__}: {exc}'
        applied = _promote(record, PromoteReason.TICKET_FAILED, ports, context, cause=cause)
        return replace(applied, curator_unreachable=True)
    match parse_submit_task_result(response):
        case SubmittedTicket(ticket_id=ticket):
            ports.queue.stamp_triage(
                record.id, triaged_by=ROUTER_ID, triage_note=hold_note(ticket, datetime.now(UTC)),
            )
            return Applied(record.id, CONVERT_PENDING, ticket=ticket)
        case SubmittedTask(task_id=task_id):
            return close_converted(record, task_id, None, ports, context)
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


def close_converted(
    record: Escalation, task_id: str, ticket: str | None, ports: RouterPorts,
    context: RouteContext,
) -> Applied:
    via = f' via curator ticket {ticket}' if ticket else ''
    _close(record, CONVERTED_CLASS, f': converted:{task_id}{via}', ports, context)
    return Applied(record.id, CONVERTED_CLASS, ticket=ticket)


def _promote(
    record: Escalation, reason: PromoteReason, ports: RouterPorts, context: RouteContext,
    *, cause: str = '',
) -> Applied:
    """The D5 fallback: one L1 hinting conversion, then the L0 dismissed citing it."""
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
    ports.queue.submit(l1)
    try:
        ports.queue.resolve(
            record.id,
            f'Promoted by the info-L0 router ({context.by}) to {l1.id}: {reason}',
            dismiss=True, resolved_by=ROUTER_ID,
        )
    except Exception:
        logger.exception(
            'info-l0-router: filed %s for %s but could not dismiss the L0', l1.id, record.id,
        )
    return Applied(record.id, PROMOTED, reason=str(reason), promoted_to=l1.id)


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
    disposition: Disposition, record: Escalation, ports: RouterPorts, context: RouteContext,
) -> Applied:
    if isinstance(disposition, PromoteWithHint):
        return Applied(record.id, ROUTER_ERROR, reason=str(disposition.reason))
    try:
        return _promote(
            record, PromoteReason.ROUTER_ERROR, ports, context,
            cause=f'applying {disposition!r} failed',
        )
    except Exception:
        logger.exception(
            'info-l0-router: could not file the router_error L1 for %s (by=%s)',
            record.id, context.by,
        )
        return Applied(record.id, ROUTER_ERROR, reason=str(PromoteReason.ROUTER_ERROR))


def _emit(applied: Applied, record: Escalation, ports: RouterPorts, context: RouteContext) -> None:
    if ports.event_store is None:
        return
    try:
        ports.event_store.emit(
            EventType.info_l0_dispositioned,
            task_id=record.task_id,
            data={
                'escalation_id': record.id,
                'class': applied.disposition_class,
                'by': context.by,
                'exit_kind': str(context.exit_kind),
                'ticket': applied.ticket,
                'task_id': record.task_id,
                'decided_at': datetime.now(UTC).isoformat(),
                'reason': applied.reason,
                'promoted_to': applied.promoted_to,
            },
        )
    except Exception:
        logger.warning('info-l0-router: event emit failed for %s', record.id, exc_info=True)
