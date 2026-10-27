"""The offline lane's escalation records, filed against one sentinel task id per command.

Four kinds share a command's sentinel: the unlanded red, the stuck filing, and the two
probe records (an outage, a broken probe; PRD ``plans/offline-lane-red-path-prd.md`` D7,
section 5.10).  The lane tells its records apart only by the ids it stores.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from datetime import datetime

from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.offline_lane_records import ProbeResult, command_key

LANE_ESCALATION_ROLE = 'orchestrator-offline-lane'
OFFLINE_LANE_SENTINEL_PREFIX = '__offline_lane__'
UNLANDED_RED_CATEGORY = 'dependency_discovered'
STUCK_FILING_CATEGORY = 'infra_issue'
OUTAGE_CATEGORY = 'infra_issue'
PROBE_BROKEN_CATEGORY = 'infra_issue'
PENDING_ESCALATION = 'pending'

_SUMMARY_LIMIT = 200
_SEVERITY = 'blocking'
_LEVEL = 1
_HOLDER_LEVEL = 2
_STILL_PENDING = frozenset({'amended', 'repeat_framing'})
_NO_LONGER_PENDING = frozenset({'not_found', 'not_pending'})


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


@dataclass(frozen=True)
class ProbeReport:
    """One outage-probe answer as a record states it; ``record_dir`` is None for a probe the tick ran."""

    command: str
    probe_command: str
    cwd: str
    timeout_secs: int
    probe: ProbeResult
    record_dir: str | None


def _broken_how(report: ProbeReport) -> str:
    probe = report.probe
    if probe.returncode is None:
        return (
            f'no exit status (timed out after {report.timeout_secs} s, or could not start; '
            'the lane log says which)'
        )
    if probe.returncode < 0:
        return f'killed by signal {-probe.returncode}'
    if not probe.first_line:
        return f'exited {probe.returncode} with no first stdout line'
    return f'exited {probe.returncode}'


def _probe_lines(report: ProbeReport) -> str:
    where = report.record_dir or "probe re-run alone on the lane's poll tick"
    return (
        f'Probe: {report.probe_command}\n'
        f'Probe cwd: {report.cwd}\n'
        f'Pass: {where}\n'
        f'Run it by hand: cd {shlex.quote(report.cwd)} && {report.probe_command}'
    )


def _outage_summary(report: ProbeReport) -> str:
    return f'Offline lane [{report.command}] outage: {report.probe.failed_dependency}'[:_SUMMARY_LIMIT]


def _outage_detail(report: ProbeReport) -> str:
    return (
        f'Offline-lane command {report.command!r} went red, and its outage probe says a dependency '
        f'its tests use is not usable.\n\n'
        f'Failed dependency: {report.probe.failed_dependency}\n'
        f'{_probe_lines(report)}\n\n'
        'The lane filed nothing for these reds, and will close this record itself when the probe passes.'
    )


def _broken_summary(report: ProbeReport) -> str:
    return f'Offline lane [{report.command}] outage probe is broken: {_broken_how(report)}'[:_SUMMARY_LIMIT]


def _broken_detail(report: ProbeReport) -> str:
    return (
        f'Offline-lane command {report.command!r} went red, and its outage probe answered neither '
        f'"up" nor an outage naming a dependency.\n\n'
        f'How it broke: {_broken_how(report)}\n'
        f'{_probe_lines(report)}\n\n'
        "This command's reds are being confirmed and filed as if no probe were declared. "
        'The lane closes this record itself when a probe next exits 0, or answers an outage naming a dependency.'
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


def file_outage(queue: EscalationQueue, report: ProbeReport) -> str:
    return _file(queue, report.command, OUTAGE_CATEGORY, _outage_summary(report), _outage_detail(report))


def file_probe_broken(queue: EscalationQueue, report: ProbeReport) -> str:
    return _file(queue, report.command, PROBE_BROKEN_CATEGORY, _broken_summary(report), _broken_detail(report))


def _amended(queue: EscalationQueue, escalation_id: str, summary: str, detail: str) -> bool:
    """True while the record is pending; False when it was resolved and archived, or dismissed."""
    status = queue.amend(escalation_id, summary=summary, detail=detail, agent_role=LANE_ESCALATION_ROLE)['status']
    if status in _STILL_PENDING:
        return True
    if status in _NO_LONGER_PENDING:
        return False
    raise ValueError(f'{escalation_id}: amend refused a framing the lane wrote: {status}')


def amend_outage(queue: EscalationQueue, escalation_id: str, report: ProbeReport) -> bool:
    return _amended(queue, escalation_id, _outage_summary(report), _outage_detail(report))


def amend_probe_broken(queue: EscalationQueue, escalation_id: str, report: ProbeReport) -> bool:
    return _amended(queue, escalation_id, _broken_summary(report), _broken_detail(report))


def release_lane_record(queue: EscalationQueue, escalation_id: str, resolution: str) -> None:
    """Resolve the lane's own record, then the level-2s that hold it by membership.

    A level-2 whose only member is the record is resolved; one with other members is
    amended to say the record closed.  Idempotent: a repeat resolves and amends nothing new.
    """
    if escalation_state(queue, escalation_id) == PENDING_ESCALATION:
        queue.resolve(escalation_id, resolution, resolved_by=LANE_ESCALATION_ROLE)
    for holder in queue.get_pending():
        if holder.level != _HOLDER_LEVEL or escalation_id not in holder.members:
            continue
        if list(dict.fromkeys(holder.members)) == [escalation_id]:
            queue.resolve(holder.id, resolution, resolved_by=LANE_ESCALATION_ROLE)
        else:
            queue.amend(
                holder.id,
                detail=f'The offline lane closed its record {escalation_id}: {resolution}',
                agent_role=LANE_ESCALATION_ROLE,
            )


def escalation_state(queue: EscalationQueue, escalation_id: str) -> str | None:
    escalation = queue.get(escalation_id)
    return escalation.status if escalation is not None else None
