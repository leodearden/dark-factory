"""The offline lane's escalation records, filed against one sentinel task id per command."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.offline_lane_records import command_key

LANE_ESCALATION_ROLE = 'orchestrator-offline-lane'
OFFLINE_LANE_SENTINEL_PREFIX = '__offline_lane__'
UNLANDED_RED_CATEGORY = 'dependency_discovered'
STUCK_FILING_CATEGORY = 'infra_issue'

_SUMMARY_LIMIT = 200
_SEVERITY = 'blocking'
_LEVEL = 1


def sentinel_task_id(command: str) -> str:
    return OFFLINE_LANE_SENTINEL_PREFIX + command_key(command)


@dataclass(frozen=True)
class UnlandedRed:
    command: str
    covering_task_id: str
    task_status: str
    nodes: tuple[str, ...]
    filed_at: datetime
    record_dirs: tuple[str, ...]


@dataclass(frozen=True)
class StuckFiling:
    command: str
    nodes: tuple[str, ...]
    due_since: datetime
    reason: str


def _bulleted(items: tuple[str, ...]) -> str:
    return '\n'.join(f'  - {item}' for item in items) or '  (none)'


def _unlanded_summary(red: UnlandedRed) -> str:
    return (
        f'Offline lane [{red.command}] still red: fix task {red.covering_task_id} '
        f'({red.task_status}) has not landed its fix'
    )[:_SUMMARY_LIMIT]


def _unlanded_detail(red: UnlandedRed) -> str:
    return (
        f'Offline-lane command {red.command!r} reproduced {len(red.nodes)} test(s) on a counted '
        f'pass long after its covering task was filed, and that task has not landed a contained fix.\n\n'
        f'Covering task: {red.covering_task_id}\n'
        f'Task status now: {red.task_status}\n'
        f'Filed at: {red.filed_at.isoformat()}\n\n'
        f'Tests still red:\n{_bulleted(red.nodes)}\n\n'
        f'Run records:\n{_bulleted(red.record_dirs)}'
    )


def _stuck_summary(stuck: StuckFiling) -> str:
    return (
        f'Offline lane [{stuck.command}] cannot file a fix task for {len(stuck.nodes)} test(s): '
        f'{stuck.reason}'
    )[:_SUMMARY_LIMIT]


def _stuck_detail(stuck: StuckFiling) -> str:
    return (
        f'Offline-lane command {stuck.command!r} has had a fix-task filing due since '
        f'{stuck.due_since.isoformat()} and every retry has failed.\n\n'
        f'Reason: {stuck.reason}\n\n'
        f'Tests waiting for a task:\n{_bulleted(stuck.nodes)}'
    )


def _file(queue: EscalationQueue, command: str, category: str, summary: str, detail: str) -> str:
    sentinel = sentinel_task_id(command)
    return queue.submit(
        Escalation(
            id=queue.make_id(sentinel),
            task_id=sentinel,
            agent_role=LANE_ESCALATION_ROLE,
            severity=_SEVERITY,
            category=category,
            summary=summary,
            detail=detail,
            level=_LEVEL,
        )
    )


def file_unlanded_red(queue: EscalationQueue, red: UnlandedRed) -> str:
    return _file(
        queue, red.command, UNLANDED_RED_CATEGORY, _unlanded_summary(red), _unlanded_detail(red)
    )


def file_stuck_filing(queue: EscalationQueue, stuck: StuckFiling) -> str:
    return _file(
        queue, stuck.command, STUCK_FILING_CATEGORY, _stuck_summary(stuck), _stuck_detail(stuck)
    )


def amend_stuck_filing(queue: EscalationQueue, escalation_id: str, stuck: StuckFiling) -> None:
    queue.amend(
        escalation_id,
        summary=_stuck_summary(stuck),
        detail=_stuck_detail(stuck),
        agent_role=LANE_ESCALATION_ROLE,
    )


def resolve_stuck_filing(queue: EscalationQueue, escalation_id: str) -> None:
    queue.resolve(
        escalation_id,
        'No filing of this command has stayed due; the lane closed its own record.',
        resolved_by=LANE_ESCALATION_ROLE,
    )


def escalation_state(queue: EscalationQueue, escalation_id: str) -> str | None:
    escalation = queue.get(escalation_id)
    return escalation.status if escalation is not None else None
