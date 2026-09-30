"""The pure decision half of the info-L0 disposition router
(plans/info-l0-disposition-router-prd.md D7/D8/§Contract).

``route_info_l0`` maps one open info-severity L0 record, plus the exit context
it is routed at, to one frozen ``Disposition`` variant.  The applier that
performs the queue writes, curator tickets and events is δ's, at
orchestrator/src/orchestrator/escalation_router.py.  Pure means no I/O and no
mutation, so the verdict table can be tested exhaustively.
"""

from __future__ import annotations

import re
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, assert_never

from escalation.classify import info_l0_mechanical_class
from escalation.models import Escalation

__all__ = [
    'Addressed',
    'ConvertViaCurator',
    'CuratorCandidate',
    'Defer',
    'DeferReason',
    'Disposition',
    'ExitKind',
    'ObservationConsumed',
    'PromoteReason',
    'PromoteWithHint',
    'ReviewerDisposition',
    'StatusInfo',
    'route_info_l0',
]


class ExitKind(StrEnum):
    """The workflow exit, sweep or restart at which an info L0 is routed."""

    DONE = 'done'
    MERGE_DEFERRED = 'merge_deferred'
    BLOCKED = 'blocked'
    CANCELLED = 'cancelled'
    REQUEUED = 'requeued'
    ORPHAN = 'orphan'
    RESTART = 'restart'


class ReviewerDisposition(StrEnum):
    """The reviewer's verdict on one info note."""

    ADDRESSED = 'addressed'
    NO_ACTION = 'no_action'
    WORK = 'work'


class PromoteReason(StrEnum):
    # NO_CONVERTIBLE_CONTENT is this module's chain terminus; the rest are δ's applier and hold-finisher reasons (PRD D5/D12).
    NO_CONVERTIBLE_CONTENT = 'no_convertible_content'
    TICKET_FAILED = 'ticket_failed'
    TICKET_REFUSED = 'ticket_refused'
    TICKET_TIMEOUT = 'ticket_timeout'
    CONVERSION_CAP = 'conversion_cap'
    ROUTER_ERROR = 'router_error'


class DeferReason(StrEnum):
    REQUEUED_EXIT = 'requeued_exit'


@dataclass(frozen=True)
class CuratorCandidate:
    """The content of a follow-up task the curator judges; transport keys are the applier's."""

    title: str
    description: str
    escalation_id: str
    spawned_from: str


@dataclass(frozen=True)
class Addressed:
    resolution_class: ClassVar[str] = 'addressed'


@dataclass(frozen=True)
class ObservationConsumed:
    resolution_class: ClassVar[str] = 'observation-consumed'


@dataclass(frozen=True)
class StatusInfo:
    class_key: str
    resolution_class: ClassVar[str] = 'status-info'


@dataclass(frozen=True)
class ConvertViaCurator:
    payload: CuratorCandidate


@dataclass(frozen=True)
class PromoteWithHint:
    reason: PromoteReason


@dataclass(frozen=True)
class Defer:
    reason: DeferReason


Disposition = Addressed | ObservationConsumed | StatusInfo | ConvertViaCurator | PromoteWithHint | Defer

# Reads the author's own declaration, not prose: 48.5% of agent info L0s open
# suggested_action with one, while detail never leads with it (archive, n=966).
_NO_ACTION_DECLARATION = re.compile(
    r'\s*(no\s+(further\s+)?action\b|none\s+(required|needed)\b|nothing\s+(to\s+do|required|needed)\b)',
    re.IGNORECASE,
)


def route_info_l0(
    record: Escalation,
    *,
    task_status: str | None,
    exit_kind: ExitKind,
    mechanical_roles: AbstractSet[str],
    reviewer_disposition: ReviewerDisposition | None = None,
) -> Disposition:
    """Decide the disposition of one open info-severity L0 *record*.

    Precedence, first match wins (rationale: the PRD's D7 and D8):

    1. A requeued exit defers the note to the next incarnation's reviewer.
    2. A reviewer disposition decides: addressed, no_action or work.
    3. A mechanical record closes as status-info, keyed by its class.
    4. An author's no-action declaration opening suggested_action is consumed.
    5. A note with a summary or detail is handed to the curator.
    6. Otherwise it is promoted with the no-convertible-content hint.

    Raises ValueError unless *record* is info severity at level 0.
    """
    _require_info_l0(record)
    if exit_kind == ExitKind.REQUEUED:
        return Defer(DeferReason.REQUEUED_EXIT)
    if reviewer_disposition is not None:
        return _reviewer_verdict(record, reviewer_disposition, task_status, exit_kind)
    class_key = info_l0_mechanical_class(record, mechanical_roles)
    if class_key is not None:
        return StatusInfo(class_key)
    if _author_declares_no_action(record):
        return ObservationConsumed()
    return _convert_or_promote(record, task_status, exit_kind)


def _require_info_l0(record: Escalation) -> None:
    severity = str(record.severity or '').strip().lower()
    if severity != 'info' or record.level != 0:
        raise ValueError(
            f'route_info_l0 routes info-severity L0s only; {record.id!r} has '
            f'severity={record.severity!r} level={record.level!r}'
        )


def _reviewer_verdict(
    record: Escalation,
    verdict: ReviewerDisposition,
    task_status: str | None,
    exit_kind: ExitKind,
) -> Disposition:
    match verdict:
        case ReviewerDisposition.ADDRESSED:
            return Addressed()
        case ReviewerDisposition.NO_ACTION:
            return ObservationConsumed()
        case ReviewerDisposition.WORK:
            return _convert_or_promote(record, task_status, exit_kind)
        case _:
            assert_never(verdict)


def _author_declares_no_action(record: Escalation) -> bool:
    return _NO_ACTION_DECLARATION.match(record.suggested_action or '') is not None


def _convert_or_promote(
    record: Escalation, task_status: str | None, exit_kind: ExitKind,
) -> ConvertViaCurator | PromoteWithHint:
    candidate = _curator_candidate(record, task_status, exit_kind)
    if candidate is None:
        return PromoteWithHint(PromoteReason.NO_CONVERTIBLE_CONTENT)
    return ConvertViaCurator(candidate)


def _curator_candidate(
    record: Escalation, task_status: str | None, exit_kind: ExitKind,
) -> CuratorCandidate | None:
    """The curator's view of *record*, or None when summary and detail are both blank."""
    summary = record.summary or ''
    detail = record.detail or ''
    title = summary.strip() or _first_non_blank_line(detail)
    if not title:
        return None
    description = '\n'.join([
        f'Escalation: {record.id}',
        f'Filed by: {record.agent_role}',
        f'Subject task: {record.task_id}',
        f'Subject status: {task_status or "unknown"}',
        f'Exit kind: {exit_kind}',
        '',
        f'Summary: {summary}',
        f'Suggested action: {record.suggested_action or ""}',
        'Detail:',
        detail,
    ])
    return CuratorCandidate(
        title=title,
        description=description,
        escalation_id=record.id,
        spawned_from=record.task_id,
    )


def _first_non_blank_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), '')
