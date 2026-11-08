#!/usr/bin/env python3
"""Write the cross-project return brief Leo pulls on return: what needs him, then what the fleet did while he was away.

The header, then a Host block, then six sections in the order the task
mandates. The header states the generation stamp, the window, the queues
scanned, the project roots measured and where that list came from (with every
scanned queue outside them), the preparation's freshness, the preparer trial,
and the sitting trigger evaluator's last run (``sitting.trigger_state``). The
Host block lists every pending record a host filer raised
(``authority.HOST_FILER_ROLES``), whatever its level, before anything else.
Section 1 is the prepare-sitting brief, always recommend-only, rendered by
``brief.render_brief`` so one numbered-entry renderer serves both documents,
followed by the "Unanswered since" list: each numbered item, oldest first,
with the stamp its ledger entry was first presented at. Sections 2-6, the Host
block and the header's figures are ``fleet_state`` measurements, each printed
with its own stamp. A store that cannot be read is a stated shortfall in its
section; no section is ever omitted.

Sections 2 and 6 stay apart on purpose. Section 2 is the standing-policy
adjudicator's shadow agreement; section 6 is this preparer's own carve-out
closes, read back from ``closing_evidence``. Leo audits the two mechanisms
independently. Section 6 also carries the executor audit draw
(``sitting.audits``): over the trailing week, whatever ``--window-days`` is,
each unaudited close of a class still below its full-audit count plus a
random ten seeded by the render date, numbered into the same ledger as
``audit N`` items the ``apply.py audit`` verb answers.

The night's numbering is a fresh sitting seeded from the previous night's
``--ledger-out`` and written back there, so an item keeps its number from night
to night, and an item that left since the last render is listed under Done with
the number it had. A watcher seeds its session ledger from it
(``prepare_sitting.py new-sitting --seed``), so the numbers Leo reads here are
the numbers his session applies. That holds for a session launched on an
earlier night too, for every item a render had already numbered by then.

No subprocess, no network, no git. Exit codes: 0 whenever the page is written,
degraded or not; 2 on a configuration error.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

# Bind scripts/ and orchestrator/ to THIS checkout, never an editable install's (tasks 2881/2882).
# Bare sys.path calls, the one statement ruff's E402 lets an import follow.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'orchestrator' / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from _task_db_scan import discover_project_roots
from orchestrator.digest import ModelRoleRow
from orchestrator.session_registry import decisions_dir, fleet_root, sessions_dir
from shared.project_layout import KNOWN_PROJECT_ROOTS_ENV, known_project_roots
from shared.safe_io import atomic_write_text
from sitting import audits, brief, fleet_state, prepare_sitting, trigger_state
from sitting import ledger as ledgers
from sitting.audits import AuditDraw, ExecutorClose
from sitting.fleet_state import (
    AutonomousClose,
    AutonomousCloses,
    HostAlarm,
    Measurement,
    PreparerTrial,
    ProjectState,
    StandingPolicyRulings,
    StuckRow,
    Window,
)
from sitting.inventory import (
    AUDIT_KIND,
    Shortfall,
    age_days,
    audit_key,
    key_str,
    parse_stamp,
    queue_project_root,
)
from sitting.prepare_sitting import Sitting
from sitting.trigger_state import TriggerLastRun

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = _REPO_ROOT / 'data' / 'return-brief.md'
DEFAULT_LEDGER_OUT = _REPO_ROOT / 'data' / 'sitting' / 'ledger-nightly.json'
DEFAULT_PREPARATION = prepare_sitting.DEFAULT_PREPARATION
DEFAULT_TRIGGER_STATE = trigger_state.DEFAULT_PATH
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

NO_EXECUTOR_CLOSE = f'no executor close in window (the trailing {audits.AUDIT_WINDOW_DAYS} days); nothing to audit'
ALL_AUDITED = 'every executor close in the window is audited'
AGED_OUT = ('{count} close(s) in the window name an escalation that has aged out of the archive; '
            'their class cannot be read')


@dataclass(frozen=True)
class PreparationFreshness:
    """How current the nightly Fable run's judgement is, so a failed night is visible on the page."""

    newest_prepared_at: str | None
    awaiting: int
    numbered: int


@dataclass(frozen=True)
class RootsStatement:
    """The project roots the per-project sections measured, where that list came from, and each queue outside them."""

    roots: tuple[str, ...]
    source: str
    outside: tuple[str, ...]


@dataclass(frozen=True)
class UnansweredRow:
    number: int
    record_id: str
    first_presented_at: str
    age_days: float | None


TriggerReading = TriggerLastRun | Shortfall | None
"""The evaluator's last run; None when it has never run; a Shortfall when its file cannot be read."""


@dataclass(frozen=True)
class ReturnBriefDocument:
    generated_at: str
    window: Window
    roots: RootsStatement
    trigger: TriggerReading
    host_alarms: Measurement[tuple[HostAlarm, ...]]
    sitting: Sitting
    preparation: PreparationFreshness
    projects: Mapping[str, ProjectState]
    closes: Measurement[AutonomousCloses]
    audits: Measurement[AuditDraw]


def build(
    *,
    project_roots: Sequence[str],
    roots_source: str,
    decisions_root: Path,
    sessions_root: Path,
    handover_override: prepare_sitting.HandoverOverride | None,
    preparation_path: Path,
    previous_ledger: Path | None,
    trigger_state_path: Path,
    window: Window,
    now: datetime,
) -> ReturnBriefDocument:
    """Read every store once, number the open items as a recommend-only sitting seeded from *previous_ledger*, and measure the fleet."""
    sources = prepare_sitting.read_sources(
        project_roots=project_roots, decisions_root=decisions_root, sessions_root=sessions_root,
        handover_override=handover_override, preparation_path=preparation_path, project=None, now=now,
    )
    seed, unseeded = _previous_numbering(previous_ledger)
    sitting = prepare_sitting.compose(
        sources, ledgers.new_sitting(now.isoformat(), seed=seed), recommend_only=True, multi_sitting=False, now=now,
    )
    if unseeded is not None:
        sitting = replace(sitting, shortfalls=(*sitting.shortfalls, unseeded))
    draw_seed = now.date().isoformat()
    audit_draw = fleet_state.unreadable_if_raises(
        'decision_registry', str(decisions_dir(decisions_root)), _no_draw(draw_seed),
        lambda: _audit_draw(decisions_root, sources.inventory.escalation_index, draw_seed, now), now=now,
    )
    if audit_draw.ok:
        sitting = replace(sitting, ledger=_number_audits(sitting.ledger, audit_draw.value, now))
    return ReturnBriefDocument(
        generated_at=now.isoformat(),
        window=window,
        roots=_roots_statement(project_roots, roots_source, sources.queues_scanned),
        trigger=_read_trigger(trigger_state_path),
        host_alarms=fleet_state.host_alarms(sources.inventory.pending, now=now),
        sitting=sitting,
        preparation=_freshness(sources, sitting),
        projects=fleet_state.measure_projects(project_roots, sources.inventory.pending, window=window, now=now),
        closes=fleet_state.unreadable_if_raises(
            'decision_registry', str(decisions_dir(decisions_root)), AutonomousCloses((), (), 0),
            lambda: fleet_state.autonomous_closes(decisions_root, window, now=now), now=now,
        ),
        audits=audit_draw,
    )


def render(document: ReturnBriefDocument) -> str:
    heading = {slug: f'{number}. {title}' for number, (slug, title) in enumerate(SECTIONS, start=1)}
    measured = _measurements(document)
    projects = list(document.projects)
    return '\n'.join([
        _header(document),
        _host_section(document.host_alarms),
        _decisions_section(document.sitting, heading['decisions']),
        _per_project_section(heading['standing_policy'], projects, measured['standing_policy'], _agreement_lines,
                             preface=_standing_policy_preface(document)),
        _per_project_section(heading['landed'], projects, measured['landed'], _landed_lines),
        _per_project_section(heading['stuck'], projects, measured['stuck'], _stuck_lines),
        _per_project_section(heading['spend'], projects, measured['spend'], _spend_lines),
        _closes_section(document.closes, heading['closes']),
        _audit_draw_section(document.audits, document.sitting.ledger),
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
            roots_source=_roots_source(args.project_roots),
            decisions_root=fleet_root(args.decisions_root),
            sessions_root=args.sessions_root if args.sessions_root is not None else sessions_dir(),
            handover_override=args.handover,
            preparation_path=args.preparation,
            previous_ledger=args.ledger_out,
            trigger_state_path=args.trigger_state,
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


def _roots_source(command_line_roots: Sequence[str] | None) -> str:
    """Which rung of ``_task_db_scan.resolve_project_roots``'s precedence supplied the roots."""
    if command_line_roots:
        return '--project-root'
    return KNOWN_PROJECT_ROOTS_ENV if known_project_roots() else 'default'


def _roots_statement(project_roots: Sequence[str], source: str, queues_scanned: Sequence[str]) -> RootsStatement:
    measured = {Path(root).resolve() for root in project_roots}
    outside = tuple(queue for queue in queues_scanned if queue_project_root(queue) not in measured)
    return RootsStatement(tuple(project_roots), source, outside)


def _read_trigger(path: Path) -> TriggerReading:
    try:
        return trigger_state.load(path)
    except trigger_state.TriggerStateUnreadable as exc:
        return Shortfall('trigger_state', str(exc.path), exc.reason)


def _previous_numbering(path: Path | None) -> tuple[ledgers.Ledger | None, Shortfall | None]:
    """The ledger the last render wrote, or why tonight numbers afresh despite one being there."""
    if path is None:
        return None, None
    try:
        return ledgers.load(path), None
    except ledgers.LedgerCorrupt as exc:
        return None, Shortfall('nightly_ledger', str(exc.path), f'{exc.reason}; numbered afresh')


def _no_draw(seed: str) -> AuditDraw:
    return AuditDraw(state='no_executor_close', full=(), sampled=(), seed=seed, population=0, aged_out=0)


def _audit_draw(
    decisions_root: Path, escalation_index: Mapping[str, Mapping[str, Path]], seed: str, now: datetime,
) -> Measurement[AuditDraw]:
    """The week's executor closes drawn for audit; a join that fails is a shortfall on an ok draw."""
    registry = decisions_dir(decisions_root)
    if not registry.is_dir():
        missing = Shortfall('decision_registry', str(registry), 'no such directory')
        return Measurement(_no_draw(seed), now.isoformat(), str(registry), 'source_missing', (missing,))
    week = Window.trailing(now, days=audits.AUDIT_WINDOW_DAYS)
    closes = audits.executor_closes(decisions_root, escalation_index, week)
    drawn = audits.draw(closes.in_window, seed=seed, audited_by_class=closes.audited_by_class,
                        aged_out=closes.aged_out)
    return Measurement(drawn, now.isoformat(), str(registry), 'ok', closes.shortfalls)


def _number_audits(ledger: ledgers.Ledger, draw: AuditDraw, now: datetime) -> ledgers.Ledger:
    """Number the drawn closes as ``audit`` items, full-audit first; an audit no longer drawn goes done."""
    order = {
        key_str(audit_key(close.decision_id)): (rank, close.closed_at)
        for rank, closes in enumerate((draw.full, draw.sampled))
        for close in closes
    }
    return ledgers.assign(ledger, order, order.__getitem__, now.isoformat(), kinds=frozenset({AUDIT_KIND}))


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
        'closes': (document.closes, document.audits),
    }


def _header(document: ReturnBriefDocument) -> str:
    window = document.window
    trial_lines = [f'  - {_trial_text(project, state.trial)}' for project, state in document.projects.items()]
    return _text([
        '# Return brief', '',
        f'- generated_at: {document.generated_at}',
        f'- window: {window.start.isoformat()} to {window.end.isoformat()}',
        f'- queues scanned: {", ".join(document.sitting.queues_scanned) or "none"}',
        f'- project roots (source: {document.roots.source}): {", ".join(document.roots.roots) or "none"}',
        f'- queues scanned outside those roots: {", ".join(document.roots.outside) or "none"}',
        f'- preparation: {_freshness_text(document.preparation, document.generated_at)}',
        '- preparer trial (recommend-only; agreement counts Leo choosing the recommended option):',
        *(trial_lines or ['  - no project root was measured']),
        f'- sitting trigger: {_trigger_text(document.trigger, document.generated_at)}',
        f'- supersedes `{SUPERSEDED_DIGEST}`: this page is the pull-on-return surface.',
        *(_shortfall_lines([document.trigger]) if isinstance(document.trigger, Shortfall) else []),
    ])


def _trigger_text(trigger: TriggerReading, generated_at: str) -> str:
    if trigger is None:
        return 'evaluator has not run'
    if isinstance(trigger, Shortfall):
        return f'unreadable — {trigger.reason}'
    lane = f' (lane {trigger.lane})' if trigger.lane else ''
    return (f'last ran {trigger.ran_at} ({_hours_apart(trigger.ran_at, generated_at)}); '
            f'outcome {trigger.outcome}: {trigger.reason}; last launch {trigger.last_launch_at or "never"}{lane}')


def _host_section(measurement: Measurement[tuple[HostAlarm, ...]]) -> str:
    lines = ['## Host', '', _stamp_line(measurement), '']
    alarms = measurement.value
    lines += [_host_line(alarm) for alarm in alarms] or ['No host alarm pending.']
    return _text([*lines, '', *_shortfall_lines(measurement.shortfalls)])


def _host_line(alarm: HostAlarm) -> str:
    age = 'age unknown' if alarm.age_days is None else f'{alarm.age_days:.1f}d old'
    return (f'- {alarm.escalation_id} ({alarm.project or alarm.queue_dir}) L{alarm.level} {alarm.severity}, '
            f'{age}, filed by {alarm.agent_role}: {alarm.summary}')


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
    unanswered = [_unanswered_line(row) for row in _unanswered(sitting)] or ['Nothing numbered awaits an answer.']
    return rendered + '\n' + _text([
        '### Unanswered since', '', *unanswered, '', docket, '', *_shortfall_lines(sitting.shortfalls),
    ])


def _unanswered(sitting: Sitting) -> tuple[UnansweredRow, ...]:
    """Every numbered question item with the stamp its ledger entry was first presented at, oldest first."""
    now = parse_stamp(sitting.generated_at)
    rows = []
    for entry in sitting.numbered:
        presented = sitting.ledger.entries[key_str(entry.item.key)].first_presented_at
        age = age_days(presented, now) if now is not None else None
        rows.append(UnansweredRow(entry.number, entry.item.key[-1], presented, age))
    return tuple(sorted(rows, key=lambda row: (parse_stamp(row.first_presented_at) or now, row.number)))


def _unanswered_line(row: UnansweredRow) -> str:
    age = 'age unknown' if row.age_days is None else f'{row.age_days:.1f}d unanswered'
    return f'- {row.number} ({row.record_id}): first presented {row.first_presented_at}, {age}'


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
            lines += ['', f'### Close {position}: decision {close.decision_id}{sample}', '',
                      *_close_lines(close, f'closed {close.closed_at}')]
        if closes.undated:
            lines += ['', '### Undated closes', '',
                      'closed_at is missing or unparsable, so these cannot be placed in or out of the window.']
            for close in closes.undated:
                lines += ['', f'#### Decision {close.decision_id}', '',
                          *_close_lines(close, f'closed_at {close.closed_at!r}')]
    return _text([*lines, '', *_shortfall_lines(measurement.shortfalls)])


def _audit_draw_section(measurement: Measurement[AuditDraw], ledger: ledgers.Ledger) -> str:
    """Section 6's executor audit draw: one ``audit N`` block per drawn close, or the state that drew none."""
    lines = ['### Executor audit draw', '', _stamp_line(measurement)]
    if measurement.ok:
        draw = measurement.value
        lines += ['', f'Executor closes in the trailing {audits.AUDIT_WINDOW_DAYS} days: {draw.population}; '
                      f'drawn {len(draw.full)} in full and {len(draw.sampled)} of at most {audits.SAMPLE_SIZE} '
                      f'at random (seed {draw.seed}). A `wrong` counts toward flipping its class to shadow '
                      'on its queue.']
        if draw.aged_out > 0:
            lines += ['', f'{AGED_OUT.format(count=draw.aged_out)}.']
        if draw.state == 'no_executor_close':
            lines += ['', f'Draw: {NO_EXECUTOR_CLOSE}.']
        elif draw.state == 'all_audited':
            lines += ['', f'Draw: {ALL_AUDITED}.']
        for close in draw.full:
            lines += ['', *_audit_item_lines(close, ledger, ' — FULL AUDIT (its class is below its full-audit count)')]
        for close in draw.sampled:
            lines += ['', *_audit_item_lines(close, ledger, '')]
    return _text([*lines, '', *_shortfall_lines(measurement.shortfalls)])


def _audit_item_lines(close: ExecutorClose, ledger: ledgers.Ledger, marker: str) -> list[str]:
    number = ledger.entries[key_str(audit_key(close.decision_id))].number
    return [
        f'#### audit {number}: decision {close.decision_id}{marker}', '',
        f'class {close.policy_class}; escalation {close.escalation_id} in {close.queue_dir}; '
        f'project {close.project}; closed {close.closed_at}', '',
        f'answer slot: `{audits.answer_slot(number)}` (to `apply.py audit --answer`)', '',
        brief.fence_safe(close.evidence),
    ]


def _close_lines(close: AutonomousClose, closed: str) -> list[str]:
    return [
        f'{close.state}; escalation {close.escalation_id or "none"}; project {close.project}; '
        f'{closed}; filed {close.filed_at}', '',
        brief.fence_safe(close.evidence),
    ]


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
                        help='the nightly numbering: seeded from the one here, written back, and what a watcher seeds from')
    parser.add_argument('--project-root', action='append', dest='project_roots',
                        help='repeatable; default: _task_db_scan.discover_project_roots()')
    parser.add_argument('--decisions-root', type=Path, help='fleet root holding decisions/ (default: fleet_root())')
    parser.add_argument('--sessions-root', type=Path, help='default: session_registry.sessions_dir()')
    parser.add_argument('--handover', type=prepare_sitting.parse_handover_override, metavar='PROJECT=PATH',
                        help="read PATH as PROJECT's handover (default: each project root's own)")
    parser.add_argument('--preparation', type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument('--trigger-state', type=Path, default=DEFAULT_TRIGGER_STATE,
                        help="the sitting trigger evaluator's last-run file (default: trigger_state.DEFAULT_PATH)")
    parser.add_argument('--window-days', type=_positive_days, default=1.0)
    parser.add_argument('--now', type=prepare_sitting.parse_instant, help='ISO-8601 instant to run at (default: now)')
    return parser


if __name__ == '__main__':
    sys.exit(main())
