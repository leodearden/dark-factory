#!/usr/bin/env python3
"""Write the cross-project return brief Leo pulls on return: what needs him, then what the fleet did while he was away.

Six sections in the order the task mandates. Section 1 is the prepare-sitting
brief, always recommend-only, rendered by ``brief.render_brief`` so one
numbered-entry renderer serves both documents. Sections 2-6 and the header are
``fleet_state`` measurements, each printed with its own stamp. A store that
cannot be read is a stated shortfall in its section; no section is ever omitted.

Sections 2 and 6 stay apart on purpose. Section 2 is the standing-policy
adjudicator's shadow agreement; section 6 is this preparer's own carve-out
closes, read back from ``closing_evidence``. Leo audits the two mechanisms
independently.

The night's numbering is written to ``--ledger-out`` as a fresh sitting. A
watcher seeds its session ledger from it (``prepare_sitting.py new-sitting
--seed``), so the numbers Leo reads here are the numbers his session applies.

No subprocess, no network, no git. Exit codes: 0 whenever the page is written,
degraded or not; 2 on a configuration error.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

# Bind scripts/ and orchestrator/ to THIS checkout, never an editable install's (tasks 2881/2882).
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / 'orchestrator' / 'src', _REPO_ROOT / 'scripts'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _task_db_scan import discover_project_roots  # noqa: E402
from orchestrator.digest import ModelRoleRow  # noqa: E402
from orchestrator.session_registry import fleet_root, sessions_dir  # noqa: E402
from shared.safe_io import atomic_write_text  # noqa: E402
from sitting import brief, fleet_state, prepare_sitting  # noqa: E402
from sitting import ledger as ledgers  # noqa: E402
from sitting.fleet_state import (  # noqa: E402
    AutonomousCloses,
    Measurement,
    PreparerTrial,
    ProjectState,
    StandingPolicyRulings,
    StuckRow,
    Window,
)
from sitting.inventory import Shortfall, parse_stamp  # noqa: E402
from sitting.prepare_sitting import Sitting  # noqa: E402

DEFAULT_OUTPUT = _REPO_ROOT / 'data' / 'return-brief.md'
DEFAULT_LEDGER_OUT = _REPO_ROOT / 'data' / 'sitting' / 'ledger-nightly.json'
DEFAULT_PREPARATION = prepare_sitting.DEFAULT_PREPARATION
SUPERSEDED_DIGEST = 'data/afk-digest.md'

EXIT_OK, EXIT_CONFIG = 0, 2

SECTIONS: tuple[tuple[str, str], ...] = (
    ('decisions', 'Decisions needed'),
    ('standing_policy', 'Rulings made under standing policy'),
    ('landed', 'Landed'),
    ('stuck', 'Stuck and why'),
    ('spend', 'Spend and cap hits'),
    ('closes', 'Autonomous closes'),
)
"""(slug, title) in the mandated order; the heading is ``<n>. <title>`` and the done line names each slug."""

SectionStatus = Literal['ok', 'degraded']


@dataclass(frozen=True)
class PreparationFreshness:
    """How current the nightly Fable run's judgement is, so a failed night is visible on the page."""

    newest_prepared_at: str | None
    awaiting: int
    numbered: int


@dataclass(frozen=True)
class ReturnBriefDocument:
    generated_at: str
    window: Window
    sitting: Sitting
    preparation: PreparationFreshness
    projects: Mapping[str, ProjectState]
    closes: Measurement[AutonomousCloses]


def build(
    *,
    project_roots: Sequence[str],
    decisions_root: Path,
    sessions_root: Path,
    handover_path: Path | None,
    preparation_path: Path,
    window: Window,
    now: datetime,
) -> ReturnBriefDocument:
    """Read every store once, number the open items as a fresh recommend-only sitting, and measure the fleet."""
    sources = prepare_sitting.read_sources(
        project_roots=project_roots, decisions_root=decisions_root, sessions_root=sessions_root,
        handover_path=handover_path, preparation_path=preparation_path, project=None, now=now,
    )
    sitting = prepare_sitting.compose(
        sources, ledgers.new_sitting(now.isoformat()), recommend_only=True, multi_sitting=False, now=now,
    )
    return ReturnBriefDocument(
        generated_at=now.isoformat(),
        window=window,
        sitting=sitting,
        preparation=_freshness(sources, sitting),
        projects=fleet_state.measure_projects(project_roots, sources.inventory.escalation_index,
                                              window=window, now=now),
        closes=fleet_state.autonomous_closes(decisions_root, window, now=now),
    )


def render(document: ReturnBriefDocument) -> str:
    heading = {slug: f'{number}. {title}' for number, (slug, title) in enumerate(SECTIONS, start=1)}
    measured = _measurements(document)
    projects = list(document.projects)
    return '\n'.join([
        _header(document),
        _decisions_section(document.sitting, heading['decisions']),
        _per_project_section(heading['standing_policy'], projects, measured['standing_policy'], _agreement_lines,
                             preface=_standing_policy_preface(document)),
        _per_project_section(heading['landed'], projects, measured['landed'], _landed_lines),
        _per_project_section(heading['stuck'], projects, measured['stuck'], _stuck_lines),
        _per_project_section(heading['spend'], projects, measured['spend'], _spend_lines),
        _closes_section(document.closes, heading['closes']),
    ])


def section_statuses(document: ReturnBriefDocument) -> dict[str, SectionStatus]:
    """``ok`` when every measurement behind a section answered in full, else ``degraded``."""
    clean = {slug: all(m.ok and not m.shortfalls for m in ms) for slug, ms in _measurements(document).items()}
    clean['decisions'] = not document.sitting.shortfalls
    return {slug: 'ok' if clean[slug] else 'degraded' for slug, _ in SECTIONS}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    now = args.now or datetime.now(UTC)
    try:
        document = build(
            project_roots=args.project_roots or discover_project_roots(),
            decisions_root=fleet_root(args.decisions_root),
            sessions_root=args.sessions_root if args.sessions_root is not None else sessions_dir(),
            handover_path=prepare_sitting.handover_file(args.handover),
            preparation_path=args.preparation,
            window=Window.trailing(now, days=args.window_days),
            now=now,
        )
    except ValueError as exc:
        print(f'return brief not written: {exc}', file=sys.stderr)
        return EXIT_CONFIG
    atomic_write_text(args.output, render(document), mkdir=True)
    ledgers.save(args.ledger_out, document.sitting.ledger)
    print(f'wrote {args.output} and {args.ledger_out}')
    statuses = section_statuses(document)
    print(f'done ({", ".join(f"{slug}={status}" for slug, status in statuses.items())})')
    return EXIT_OK


def _freshness(sources: prepare_sitting.Sources, sitting: Sitting) -> PreparationFreshness:
    stamps = [prep.prepared_at for prep in sources.preparations.entries.values()]
    newest = max(stamps, key=lambda stamp: parse_stamp(stamp) or datetime.min.replace(tzinfo=UTC), default=None)
    awaiting = sum(1 for entry in sitting.numbered if entry.preparation is None)
    return PreparationFreshness(newest, awaiting, len(sitting.numbered))


def _measurements(document: ReturnBriefDocument) -> dict[str, tuple[Measurement[Any], ...]]:
    """Every measurement behind each measured section; the single map both render and the statuses read."""
    states = list(document.projects.values())
    return {
        'standing_policy': tuple(state.standing_policy for state in states),
        'landed': tuple(state.landed for state in states),
        'stuck': tuple(state.stuck for state in states),
        'spend': tuple(state.spend for state in states),
        'closes': (document.closes,),
    }


def _header(document: ReturnBriefDocument) -> str:
    window = document.window
    trial_lines = [f'  - {_trial_text(project, state.trial)}' for project, state in document.projects.items()]
    return _text([
        '# Return brief', '',
        f'- generated_at: {document.generated_at}',
        f'- window: {window.start.isoformat()} to {window.end.isoformat()}',
        f'- queues scanned: {", ".join(document.sitting.queues_scanned) or "none"}',
        f'- preparation: {_freshness_text(document.preparation, document.generated_at)}',
        '- preparer trial (recommend-only; agreement counts Leo choosing the recommended option):',
        *(trial_lines or ['  - no project root was measured']),
        f'- supersedes `{SUPERSEDED_DIGEST}`: this page is the pull-on-return surface.',
    ])


def _freshness_text(freshness: PreparationFreshness, generated_at: str) -> str:
    awaiting = f'{freshness.awaiting} of {freshness.numbered} numbered items awaiting preparation'
    if freshness.newest_prepared_at is None:
        return f'no preparation recorded; {awaiting}'
    return (f'newest prepared_at {freshness.newest_prepared_at} '
            f'({_hours_apart(freshness.newest_prepared_at, generated_at)}); {awaiting}')


def _hours_apart(stamp: str, generated_at: str) -> str:
    then, now = parse_stamp(stamp), parse_stamp(generated_at)
    if then is None or now is None:
        return 'age unknown'
    hours = (now - then).total_seconds() / 3600
    return f'{abs(hours):.1f}h {"before" if hours >= 0 else "after"} this render'


def _trial_text(project: str, measurement: Measurement[PreparerTrial]) -> str:
    if not measurement.ok:
        return f'{project}: not measured — {_flag_text(measurement)}'
    trial, turns = measurement.value, measurement.value.resolution_turns
    rate = f' ({trial.rate:.0%})' if trial.rate is not None else ''
    median = 'n/a' if turns.median is None else f'{turns.median:g}'
    return (f'{project}: agreement {trial.numerator}/{trial.denominator}{rate} over n={trial.n} prepared '
            f'(no lean {trial.no_lean}, unanswered {trial.unanswered}, rejected {trial.rejected}); '
            f'resolution_turns n={turns.with_turns} of {turns.resolved} resolved L2s, median {median}, '
            f'total {turns.total}; measured {measurement.measured_at}')


def _decisions_section(sitting: Sitting, heading: str) -> str:
    rendered = brief.render_brief(
        sitting.numbered, sitting.standing, sitting.done,
        glossary=sitting.glossary, generated_at=sitting.generated_at, level=2, title=heading,
    )
    docket = (f'Docket page: recommended — {sitting.docket_reason}.' if sitting.docket_reason
              else 'Docket page: not needed; answer by number.')
    return rendered + '\n' + _text([docket, '', *_shortfall_lines(sitting.shortfalls)])


def _standing_policy_preface(document: ReturnBriefDocument) -> list[str]:
    statements = dict.fromkeys(state.standing_policy.value.statement for state in document.projects.values())
    return [
        *(line for statement in statements for line in (statement, '')),
        f'Audit sample (1 in {brief.AUDIT_SAMPLE_EVERY}): none — no autonomous ruling to sample.', '',
    ]


def _per_project_section(
    heading: str,
    projects: Sequence[str],
    measurements: Sequence[Measurement[Any]],
    body: Callable[[Any], list[str]],
    *,
    preface: Iterable[str] = (),
) -> str:
    lines = [f'## {heading}', '', *preface]
    for project, measurement in zip(projects, measurements, strict=True):
        lines += [f'### {project}', '', _stamp_line(measurement)]
        if measurement.ok:
            lines += ['', *body(measurement.value)]
        lines += ['', *_shortfall_lines(measurement.shortfalls)]
    if not projects:
        lines.append('No project root was measured.')
    return _text(lines)


def _agreement_lines(rulings: StandingPolicyRulings) -> list[str]:
    report = rulings.report
    lines = ['No shadow-stamped record was resolved in the window.']
    if report.classes:
        lines = ['| class | agreed | diverged | not comparable | non-human resolver | agreement |',
                 '|---|---|---|---|---|---|']
        lines += [f'| {c.ruling_class} | {c.agreed} | {c.diverged} | {c.not_comparable} | '
                  f'{c.non_human_resolver} | {_percent(c.agreement_rate)} |' for c in report.classes]
    return [*lines, '', f'Also resolved in the window: {report.gated_stamps} gated, {report.self_resolved} '
                        f'self-resolved, {report.rejected_stamps} with an unreadable stamp. Still pending '
                        f'(lifetime backlog): {report.unresolved_lifetime}.']


def _landed_lines(count: int) -> list[str]:
    return [f'{count} task{"" if count == 1 else "s"} done in the window.']


def _stuck_lines(rows: tuple[StuckRow, ...]) -> list[str]:
    return [f'- task {row.task_id} — {row.title}: {row.reason}' for row in rows] or ['No blocked task.']


def _spend_lines(rows: tuple[ModelRoleRow, ...]) -> list[str]:
    if not rows:
        return ['No invocation completed in the window.']
    table = ['| model | role | invocations | cost | cap-hit rate | $/done |', '|---|---|---|---|---|---|']
    table += [f'| {r.model} | {r.role} | {r.invocation_count} | {_money(r.total_cost_usd)} | '
              f'{_percent(r.cap_hit_rate)} | {_money(r.cost_per_done)} |' for r in rows]
    total = sum(r.total_cost_usd for r in rows)
    return [*table, '', f'Total: {_money(total)} over {sum(r.invocation_count for r in rows)} invocations.']


def _closes_section(measurement: Measurement[AutonomousCloses], heading: str) -> str:
    lines = [f'## {heading}', '', _stamp_line(measurement)]
    if measurement.ok:
        closes = measurement.value
        lines += ['', f'Closes carrying evidence, lifetime: {closes.lifetime} '
                      '(the wrong-close kill criterion is read per 50 of these).']
        if not closes.in_window:
            lines += ['', 'No autonomous close in the window.']
        for position, close in enumerate(closes.in_window, start=1):
            sample = f' — AUDIT SAMPLE (1 in {brief.AUDIT_SAMPLE_EVERY})' if brief.is_audit_sample(position) else ''
            lines += [
                '', f'### Close {position}: decision {close.decision_id}{sample}', '',
                f'{close.state}; escalation {close.escalation_id or "none"}; project {close.project}; '
                f'filed {close.filed_at}', '',
                brief.fence_safe(close.evidence),
            ]
    return _text([*lines, '', *_shortfall_lines(measurement.shortfalls)])


def _stamp_line(measurement: Measurement[Any]) -> str:
    return f'measured {measurement.measured_at} from {measurement.source} ({measurement.status})'


def _flag_text(measurement: Measurement[Any]) -> str:
    return f'{measurement.status}: ' + '; '.join(_shortfall_text(s) for s in measurement.shortfalls)


def _shortfall_lines(shortfalls: Iterable[Shortfall]) -> list[str]:
    return [f'- shortfall: {_shortfall_text(shortfall)}' for shortfall in shortfalls]


def _shortfall_text(shortfall: Shortfall) -> str:
    return f'{shortfall.source} {shortfall.path}: {shortfall.reason}'


def _percent(rate: float | None) -> str:
    return 'n/a' if rate is None else f'{rate:.0%}'


def _money(amount: float | None) -> str:
    return 'n/a' if amount is None else f'${amount:.2f}'


def _text(lines: list[str]) -> str:
    return '\n'.join(lines).rstrip('\n') + '\n'


def _positive_days(text: str) -> float:
    days = float(text)
    if days <= 0:
        raise argparse.ArgumentTypeError(f'--window-days must be positive, got {text}')
    return days


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='return_brief.py', description='Write the cross-project return brief.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--ledger-out', type=Path, default=DEFAULT_LEDGER_OUT,
                        help='the nightly numbering, a fresh sitting a watcher seeds from')
    parser.add_argument('--project-root', action='append', dest='project_roots',
                        help='repeatable; default: _task_db_scan.discover_project_roots()')
    parser.add_argument('--decisions-root', type=Path, help='fleet root holding decisions/ (default: fleet_root())')
    parser.add_argument('--sessions-root', type=Path, help='default: session_registry.sessions_dir()')
    parser.add_argument('--handover', type=Path, help='default: this checkout\'s newest handover file')
    parser.add_argument('--preparation', type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument('--window-days', type=_positive_days, default=1.0)
    parser.add_argument('--now', type=prepare_sitting.parse_instant, help='ISO-8601 instant to run at (default: now)')
    return parser


if __name__ == '__main__':
    sys.exit(main())
