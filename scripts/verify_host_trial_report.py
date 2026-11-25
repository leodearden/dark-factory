#!/usr/bin/env python3
"""Report whether one verify host lifted fleet landings: its on-blocks against its off-blocks.

PRD plans/verify-host-roster-and-rented-trial-prd.md task ε (§6.5); the
consumer is gate λ. The report splits [--since, --until) into the host's on and
off blocks (verify_host_trial_blocks), reading every project's runs.db and the
load samples store read-only (verify_host_trial_stores). Per block and project
it prints the mechanism measures (verify_host_trial_measures) and the allowance
expired unused (verify_host_trial_allowance). It ends with one fleet decision
line (verify_host_trial_decision). Every figure is preceded by its rule.

Runs under /usr/bin/python3 with no venv: scripts/_checkout.py binds this
checkout's workspace members. Exit 0 with a report, 2 for a usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import _checkout as _checkout
import verify_host_trial_allowance as allowance
import verify_host_trial_blocks as blocks
import verify_host_trial_decision as decision
import verify_host_trial_measures as measures
import verify_host_trial_stores as stores
from resource_report import default_samples_db
from resource_report_holds import parse_utc
from sampler.metrics import MetricSpec
from shared.project_layout import default_project_root, known_project_roots
from verify_host_trial_measures import (
    FIGURE_RULES,
    FigureId,
    Measured,
    NeverWritten,
    NoHostVerify,
    Unavailable,
)

# ---------------------------------------------------------------------------
# The report model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialRequest:
    host: str
    since: datetime
    until: datetime
    roots: tuple[Path, ...]
    samples_db: Path
    host_cost_usd: float
    seat_bill_usd: float
    resamples: int
    seed: int


@dataclass(frozen=True)
class ProjectFigures:
    project: str
    figures: Mapping[FigureId, measures.Figure[Any]]


@dataclass(frozen=True)
class BlockReport:
    number: int
    block: blocks.Block
    landings: int
    """Fleet tasks whose first landing falls in the block."""
    host_verifies: int
    """Fleet host merge_verify starts in the block."""
    cpu: measures.Figure[Any]
    expired_usd: float | None
    projects: tuple[ProjectFigures, ...]


@dataclass(frozen=True)
class TrialReport:
    request: TrialRequest
    projects: tuple[stores.ProjectRead, ...]
    first_host_verify_at: datetime | None
    lifecycle: tuple[stores.LifecycleRow, ...]
    """The host's lifecycle rows over every store read."""
    lifecycle_written: bool
    """Whether any store read holds a lifecycle row for any host."""
    blocks: tuple[BlockReport, ...]
    allowance: allowance.AllowanceReading
    cpu: stores.CpuRead
    decision: decision.Decision


_TRANSITIONS: Mapping[stores.EventType, tuple[blocks.HostState, blocks.LifecycleOrigin]] = {
    stores.EventType.VERIFY_HOST_ADMITTED: (blocks.HostState.ON, blocks.LifecycleOrigin.ADMITTED),
    stores.EventType.VERIFY_HOST_RETIRED: (blocks.HostState.OFF, blocks.LifecycleOrigin.RETIRED),
}


def _transition(row: stores.LifecycleRow) -> blocks.Transition:
    return blocks.Transition(row.at, *_TRANSITIONS[row.event_type])


def _project_name(project: stores.ProjectRead) -> str:
    return project.root.name or str(project.root)


def _count(figure: measures.Figure[Any]) -> int:
    return figure.value if isinstance(figure, Measured) else 0


def _block_report(
    number: int,
    block: blocks.Block,
    read: Sequence[stores.ProjectRead],
    host: str,
    cpu: stores.CpuRead,
    reading: allowance.AllowanceReading,
) -> BlockReport:
    projects = tuple(ProjectFigures(_project_name(p), measures.project_figures(p, host, block)) for p in read)
    return BlockReport(
        number=number,
        block=block,
        landings=sum(measures.landing_count(project, block) for project in read),
        host_verifies=sum(_count(project.figures[FigureId.HOST_VERIFIES]) for project in projects),
        cpu=measures.cpu_pressure(cpu, block),
        expired_usd=allowance.expired_in_block(reading, block),
        projects=projects,
    )


def build_report(request: TrialRequest) -> TrialReport:
    projects = tuple(stores.read_project(root, request.host, request.since, request.until) for root in request.roots)
    read = [project for project in projects if project.status is stores.ProjectStatus.READ]
    lifecycle = tuple(sorted((row for project in read for row in project.lifecycle), key=lambda row: row.at))
    first_verify = min((p.first_host_verify_at for p in read if p.first_host_verify_at is not None), default=None)
    timeline = blocks.derive_blocks(map(_transition, lifecycle), first_verify, request.since, request.until)
    cpu = stores.read_cpu_pressure(request.samples_db, request.since, request.until)
    reading = allowance.account_cycles(
        [hit for project in read for hit in project.cap_hits],
        [spend for project in read for spend in project.invocations],
        request.since,
        request.until,
    )
    block_reports = tuple(
        _block_report(number, block, read, request.host, cpu, reading) for number, block in enumerate(timeline, 1)
    )
    verdict = decision.decide(
        [
            decision.BlockLandings(b.block.state is blocks.HostState.ON, b.landings, b.block.days, b.host_verifies)
            for b in block_reports
        ],
        host_cost_usd=request.host_cost_usd,
        seat_bill_usd=request.seat_bill_usd,
        resamples=request.resamples,
        seed=request.seed,
    )
    return TrialReport(
        request, projects, first_verify, lifecycle, any(p.lifecycle_ever_written for p in read),
        block_reports, reading, cpu, verdict,
    )


# ---------------------------------------------------------------------------
# Figure words, shared by both renderers
# ---------------------------------------------------------------------------

_LABELS: Mapping[FigureId, str] = {
    FigureId.HOST_VERIFIES: 'host verifies',
    FigureId.LANDINGS_PER_DAY: 'landings per day',
    FigureId.MERGE_VERIFY_MEDIAN: 'merge_verify median wall by runner',
    FigureId.TURN_WAIT_SHARE: 'turn wait share',
    FigureId.DEFERRAL_HOURS: 'dispatch_deferred hours by gate arm',
    FigureId.LOCK_CONFLICT_REQUEUES: 'lock-conflict requeues',
    FigureId.SPEND_PER_DAY: 'agent spend per day',
    FigureId.CPU_PRESSURE: 'workstation CPU pressure',
}


def _medians(value: Mapping[str, measures.RunnerMedian]) -> str:
    runners = '; '.join(f'{runner} {m.median_secs:.0f} s (n={m.n})' for runner, m in value.items())
    return runners or 'no merge_verify started in this block'


def _turn_wait(value: measures.TurnWait) -> str:
    return (
        f'{value.share:.1%} ({value.wait_secs / 3600:.2f} h waiting of {value.slot_secs / 3600:.2f} h run-slot; '
        f'{value.unpaired_starts} unpaired task_started)'
    )


def _deferrals(value: measures.DeferralHours) -> str:
    arms = ', '.join(f'{arm} {hours:.2f} h' for arm, hours in value.hours.items()) or 'none'
    return f'{arms}; {value.unspanned} unspanned'


def _requeues(value: Mapping[str, int]) -> str:
    return ', '.join(f'{reason} {count}' for reason, count in value.items()) or 'none'


def _cpu(value: Mapping[MetricSpec, measures.CpuStat | None]) -> str:
    return '; '.join(
        f'{spec.name} no sample' if stat is None else f'{spec.name} mean {stat.mean:.2f} p90 {stat.p90:.2f} (n={stat.n})'
        for spec, stat in value.items()
    )


_FORMATS: Mapping[FigureId, Callable[[Any], str]] = {
    FigureId.HOST_VERIFIES: str,
    FigureId.LANDINGS_PER_DAY: lambda value: f'{value:.2f}',
    FigureId.MERGE_VERIFY_MEDIAN: _medians,
    FigureId.TURN_WAIT_SHARE: _turn_wait,
    FigureId.DEFERRAL_HOURS: _deferrals,
    FigureId.LOCK_CONFLICT_REQUEUES: _requeues,
    FigureId.SPEND_PER_DAY: lambda value: f'${value:.2f}/day',
    FigureId.CPU_PRESSURE: _cpu,
}

NO_HOST_VERIFY = 'no verify ran on this host'


def figure_words(figure_id: FigureId, figure: measures.Figure[Any]) -> str:
    match figure:
        case Measured(value=value):
            return _FORMATS[figure_id](value)
        case NoHostVerify():
            return NO_HOST_VERIFY
        case NeverWritten(source=source):
            return f'no {source} row ever written in this store'
        case Unavailable(reason=reason):
            return f'unavailable: {reason}'


def _plural(count: int, noun: str, plural: str | None = None) -> str:
    return f'{count} {noun}' if count == 1 else f'{count} {plural or noun + "s"}'


def _usd(value: float | None) -> str:
    return 'unknown' if value is None else f'${value:.2f}'


def lifecycle_words(report: TrialReport) -> str:
    producer = stores.SOURCES[stores.Source.LIFECYCLE].producer
    if not report.lifecycle_written:
        return (
            f'host lifecycle: no {producer} row ever written in any store read; '
            "block boundaries come from the host's first merge_verify only"
        )
    inside = sum(1 for row in report.lifecycle if report.request.since <= row.at < report.request.until)
    return f'host lifecycle: {_plural(len(report.lifecycle), f"{producer} row")} for {report.request.host}, {inside} inside the window'


def decision_words(verdict: decision.Decision) -> str:
    on = 'n/a' if verdict.on_rate is None else f'{verdict.on_rate:.2f} landings/day ({_plural(verdict.on_blocks, "block")})'
    off = 'n/a' if verdict.off_rate is None else f'{verdict.off_rate:.2f} landings/day ({_plural(verdict.off_blocks, "block")})'
    if verdict.effect is None:
        outcome = str(verdict.reason)
    elif verdict.interval is None:
        outcome = f'{verdict.effect:+.1%} ({verdict.reason})'
    else:
        interval = verdict.interval
        undefined = f', {interval.undefined_resamples} resamples undefined' if interval.undefined_resamples else ''
        outcome = f'{verdict.effect:+.1%}, 95% block-bootstrap interval {interval.low:+.1%} to {interval.high:+.1%}{undefined}'
    excluded = ', '.join(str(position + 1) for position in verdict.excluded_on_blocks)
    excluded = f'; on-blocks where no verify ran on this host, excluded: {excluded}' if excluded else ''
    return (
        f'decision: on {on} against off {off}: {outcome}{excluded}; '
        f'host cost {verdict.host_cost_share:.1%} of the seat bill'
    )


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def _project_lines(project: stores.ProjectRead) -> list[str]:
    line = f'project {_project_name(project)}: {project.status} ({project.runs_db})'
    lines = [line if project.detail is None else f'{line}: {project.detail}']
    skipped = ', '.join(f'{source} {count}' for source, count in project.skipped.items() if count)
    return lines + ([f'  unparseable rows skipped: {skipped}'] if skipped else [])


def _header_lines(report: TrialReport) -> list[str]:
    request = report.request
    first = 'none' if report.first_host_verify_at is None else report.first_host_verify_at.isoformat()
    samples = str(request.samples_db) if report.cpu.reason is None else f'unavailable: {report.cpu.reason}'
    return [
        f'verify-host trial report: host {request.host} over [{request.since.isoformat()}, {request.until.isoformat()})',
        *(line for project in report.projects for line in _project_lines(project)),
        lifecycle_words(report),
        f"host's first merge_verify start in any store: {first}",
        f'samples store: {samples}',
    ]


def _figure_lines(figure_id: FigureId, figure: measures.Figure[Any], indent: str) -> list[str]:
    return [f'{indent}rule: {FIGURE_RULES[figure_id]}', f'{indent}{_LABELS[figure_id]}: {figure_words(figure_id, figure)}']


def _block_lines(report: BlockReport) -> list[str]:
    block = report.block
    verifies = _plural(report.host_verifies, 'host verify', 'host verifies') if report.host_verifies else NO_HOST_VERIFY
    lines = [
        f'block {report.number}: {block.state} [{block.start.isoformat()}, {block.end.isoformat()}) '
        f'{block.days:.2f} days, opened by {block.opened_by} at {block.opened_at.isoformat()}',
        f'  fleet: {_plural(report.landings, "landing")}; {verifies}',
        *_figure_lines(FigureId.CPU_PRESSURE, report.cpu, '  '),
        f'  rule: {allowance.ALLOWANCE_RULE}',
        f'  allowance expired unused (estimate): {_usd(report.expired_usd)}',
    ]
    for project in report.projects:
        lines.append(f'  project {project.project}')
        lines += [line for fid, figure in project.figures.items() for line in _figure_lines(fid, figure, '    ')]
    return lines


def _allowance_lines(reading: allowance.AllowanceReading) -> list[str]:
    estimate = 'unknown: no capped cycle reset in the window' if reading.estimate_usd is None else _usd(reading.estimate_usd)
    lines = [f'rule: {allowance.ALLOWANCE_RULE}', f'allowance estimate per cycle: {estimate}']
    for cycle in reading.cycles:
        state = (
            f'capped {cycle.hours_capped:.1f} h before reset' if cycle.capped
            else f'uncapped, expired {_usd(cycle.expired_usd)}'
        )
        lines.append(f'  cycle {cycle.account} resetting {cycle.reset.isoformat()}: {state}, spend {_usd(cycle.spend_usd)}')
    if reading.unplaced_accounts:
        lines.append(f'  accounts with spend and no reset ever recorded: {", ".join(reading.unplaced_accounts)}')
    if reading.dormant_accounts:
        lines.append(f'  dormant accounts (no spend or cap near the window): {", ".join(reading.dormant_accounts)}')
    if reading.unplaceable_cap_hits:
        lines.append(f'  weekly cap hits without a reset: {reading.unplaceable_cap_hits}')
    if reading.other_caps:
        counts = ', '.join(f'{kind} {count}' for kind, count in reading.other_caps.items())
        lines.append(f'  caps that place no cycle: {counts}')
    return lines


def render_text(report: TrialReport) -> str:
    return '\n'.join([
        *_header_lines(report),
        f'rule: {blocks.BLOCK_RULE}',
        *(line for block in report.blocks for line in _block_lines(block)),
        *_allowance_lines(report.allowance),
        f'rule: {decision.DECISION_RULE}',
        decision_words(report.decision),
    ])


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MetricSpec):
        return value.name
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list | frozenset):
        return [_jsonable(item) for item in value]
    return value


_FIGURE_STATES: Mapping[type, str] = {
    Measured: 'measured', NoHostVerify: 'no_host_verify', NeverWritten: 'never_written', Unavailable: 'unavailable',
}


def figure_json(figure_id: FigureId, figure: measures.Figure[Any]) -> dict[str, Any]:
    body = {'state': _FIGURE_STATES[type(figure)], 'text': figure_words(figure_id, figure)}
    return {**body, 'value': _jsonable(figure.value)} if isinstance(figure, Measured) else body


def _project_json(project: stores.ProjectRead) -> dict[str, Any]:
    return {
        'project': _project_name(project),
        'root': str(project.root),
        'runs_db': str(project.runs_db),
        'status': project.status,
        'detail': project.detail,
        'lifecycle_ever_written': project.lifecycle_ever_written,
        'sources': _jsonable(project.sources),
    }


def _block_json(report: BlockReport) -> dict[str, Any]:
    return {
        'number': report.number,
        **_jsonable(report.block),
        'days': report.block.days,
        'landings': report.landings,
        'host_verifies': report.host_verifies,
        'cpu_pressure': figure_json(FigureId.CPU_PRESSURE, report.cpu),
        'allowance_expired_usd': report.expired_usd,
        'projects': [
            {'project': p.project, 'figures': {fid: figure_json(fid, figure) for fid, figure in p.figures.items()}}
            for p in report.projects
        ],
    }


def report_json(report: TrialReport) -> dict[str, Any]:
    request = report.request
    return {
        'request': _jsonable(request),
        'rules': {
            'blocks': blocks.BLOCK_RULE,
            'decision': decision.DECISION_RULE,
            'allowance': allowance.ALLOWANCE_RULE,
            'figures': {str(fid): rule for fid, rule in FIGURE_RULES.items()},
        },
        'projects': [_project_json(project) for project in report.projects],
        'lifecycle': {'text': lifecycle_words(report), 'ever_written': report.lifecycle_written, 'rows': _jsonable(report.lifecycle)},
        'first_host_verify_at': _jsonable(report.first_host_verify_at),
        'samples_db': {'path': str(report.cpu.samples_db), 'reason': report.cpu.reason, 'unreadable_dropped': report.cpu.unreadable_dropped},
        'blocks': [_block_json(block) for block in report.blocks],
        'allowance': _jsonable(report.allowance),
        'decision': {**_jsonable(report.decision), 'text': decision_words(report.decision)},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f'must be a positive integer: {text}')
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='verify_host_trial_report.py',
        description="One verify host's on-blocks against its off-blocks: fleet landings and mechanism measures.",
    )
    parser.add_argument('--host', required=True, help='the merge_verify runner name of the host under trial')
    parser.add_argument('--since', required=True, help='ISO date or datetime (naive is UTC)')
    parser.add_argument('--until', default=None, help='ISO date or datetime (naive is UTC; default now)')
    parser.add_argument('--host-cost-usd', type=float, required=True, help="the host's cost over the trial")
    parser.add_argument('--seat-bill-usd', type=float, required=True, help='the seat bill over the trial')
    parser.add_argument(
        '--project-root', dest='project_roots', action='append', type=Path, default=None, metavar='ROOT',
        help='a project root whose runs.db to read; repeat for each (default: DASHBOARD_KNOWN_PROJECT_ROOTS, '
        'else the main checkout)',
    )
    parser.add_argument('--samples-db', type=Path, default=None, help="the load recorder's store")
    parser.add_argument('--json', action='store_true', help='print one JSON object')
    parser.add_argument('--resamples', type=_positive_int, default=10000, help='block bootstrap draws (default 10000)')
    parser.add_argument('--seed', type=int, default=0, help='block bootstrap seed (default 0)')
    return parser


def _distinct(roots: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(root.resolve() for root in roots))


def _request(parser: argparse.ArgumentParser, args: argparse.Namespace, now: datetime) -> TrialRequest:
    try:
        since = parse_utc(args.since)
        until = now if args.until is None else parse_utc(args.until)
    except ValueError as exc:
        parser.error(f'--since/--until: expected an ISO date or datetime ({exc})')
    if until <= since:
        parser.error(f'--until {until.isoformat()} is not after --since {since.isoformat()}')
    if args.host_cost_usd < 0:
        parser.error(f'--host-cost-usd must not be negative: {args.host_cost_usd}')
    if args.seat_bill_usd <= 0:
        parser.error(f'--seat-bill-usd must be positive: {args.seat_bill_usd}')
    roots = args.project_roots or known_project_roots() or (default_project_root(),)
    return TrialRequest(
        args.host, since, until, _distinct(roots), args.samples_db or default_samples_db(),
        args.host_cost_usd, args.seat_bill_usd, args.resamples, args.seed,
    )


def main(argv: Sequence[str] | None = None, *, now: datetime | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = build_report(_request(parser, args, now or datetime.now(UTC)))
    print(json.dumps(report_json(report), indent=2) if args.json else render_text(report))
    return 0


if __name__ == '__main__':
    sys.exit(main())
