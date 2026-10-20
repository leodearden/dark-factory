"""The applier half of the info-L0 disposition router.

plans/info-l0-disposition-router-prd.md D5/D7/D8/D9/D12.  The decision half is
pure and lives in escalation/src/escalation/disposition.py::route_info_l0; this
module performs what a decision means: queue writes, curator tickets, the D9
hold a ticket leaves on its record, and the per-sweep aggregates.  Its three
callers (the workflow exit finalizer, the orphan reaper and the startup
amnesty) only wire it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from escalation.disposition import ExitKind
from escalation.models import Escalation, normalised_severity
from shared.mcp_envelope import parse_tool_result

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
