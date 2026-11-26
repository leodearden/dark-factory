#!/usr/bin/env python3
"""Read-only CLI of the task-8532 merge-lane canary readout (plans/merge-lane-policy-prd.md D12).

Reads one window of an orchestrator runs.db, main's reflog in a git checkout
and a committed extract of the orchestrator journal, computes every figure and
predicate with `merge_lane_canary`, and writes a markdown readout to --out.

Usage:
    merge_lane_canary_readout.py --window <iso>..<iso> --journal-extract PATH --out PATH
        [--project-root ROOT] [--git-root ROOT] [--laptop-host laptop] [--local-host local]

STRICTLY READ-ONLY: runs.db is opened through a ``mode=ro`` URI, git is only
read, and --out is the one file written.

LOOKBACK. Red chains, retirements, submissions, chain landings and main's
reflog are also read for the 7 days before the window, so a figure whose
cause predates the window's start still sees it.

EXIT CODES: 0 the readout was written, whatever its predicates say; 3 the
window's rows predate the D10 alpha fields, and the refusal naming them was
written instead; 2 a usage fault, an unreadable runs.db or journal extract, or
a git fault.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import merge_lane_canary as canary
import merge_lane_throughput as mlt
from merge_lane_canary import Event
from shared.project_layout import runs_db

LOOKBACK = timedelta(days=7)
GIT_TIMEOUT_SECS = 120
EXIT_REFUSED = 3
EXIT_FAULT = 2
PER_ITEM_WALL_MIN_ITEMS = 8


def _iso(moment: datetime) -> str:
    """The events table's own timestamp spelling, so a TEXT comparison is chronological."""
    return moment.astimezone(UTC).isoformat()


def _connect_ro(path: Path) -> sqlite3.Connection:
    """Open *path* read-only: a write fails loudly and a wrong path raises instead of creating a db."""
    return sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)


# ---------------------------------------------------------------------------
# runs.db
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowEvents:
    """The window's events; the ``*_lookback`` fields also cover the 7 days before it."""

    verify: list[Event]
    verify_lookback: list[Event]
    cancelled: list[Event]
    voided: list[Event]
    discards: list[Event]
    finalized: list[Event]
    finalized_lookback: list[Event]
    queued_lookback: list[Event]
    train_merged: list[Event]
    train_merged_lookback: list[Event]
    gates: list[Event]
    heartbeats: list[Event]


def load_window(conn: sqlite3.Connection, lo: datetime, hi: datetime) -> WindowEvents:
    def window(event_type: str) -> list[Event]:
        return list(mlt.load_events(conn, event_type, lo, hi))

    def lookback(event_type: str) -> list[Event]:
        return list(mlt.load_events(conn, event_type, lo - LOOKBACK, hi))

    return WindowEvents(
        verify=window('merge_verify'), verify_lookback=lookback('merge_verify'),
        cancelled=window('merge_verify_cancelled'), voided=window('verdict_voided'),
        discards=window('speculative_discard'), finalized=window('merge_finalized'),
        finalized_lookback=lookback('merge_finalized'), queued_lookback=lookback('merge_queued'),
        train_merged=window('train_merged'), train_merged_lookback=lookback('train_merged'),
        gates=window('merge_reverify_gate'), heartbeats=window('merge_heartbeat'),
    )


TaskMoment = tuple[datetime, str]


def _moments(conn: sqlite3.Connection, sql: str, params: tuple[object, ...]) -> list[TaskMoment]:
    rows = conn.execute(sql, params).fetchall()
    return [(moment, str(task)) for raw, task in rows if (moment := _parse_instant(raw)) is not None]


def _parse_instant(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


_PHASE_ROWS = (
    'SELECT timestamp, task_id FROM events WHERE event_type=? AND phase=? '
    'AND timestamp>=? AND timestamp<? ORDER BY id'
)
_GREEN_ROWS = (
    "SELECT timestamp, task_id FROM events WHERE event_type='workflow_verify' "
    "AND json_extract(data, '$.passed') = 1 AND timestamp>=? AND timestamp<? ORDER BY id"
)


def load_task_verify_starts(conn: sqlite3.Connection, lo: datetime, hi: datetime) -> list[canary.TaskVerifyStart]:
    """Task verifies started in ``[lo, hi)``: ``phase_enter`` rows with phase verify.

    The workflow emits ``workflow_verify`` (only on a green verify) just AFTER
    the verify's ``phase_exit``, so a verify passed when a passed row for its
    task is stamped after its start and before the task's next verify start.
    Without one it failed if its exit is recorded, and is undecided (None)
    otherwise. Starts, exits and verdicts are read to a day past *hi*, so a
    verify started near the end still resolves.
    """
    settle = hi + timedelta(days=1)
    enters = _moments(conn, _PHASE_ROWS, ('phase_enter', 'verify', _iso(lo), _iso(settle)))
    exits = _moments(conn, _PHASE_ROWS, ('phase_exit', 'verify', _iso(lo), _iso(settle)))
    greens = _moments(conn, _GREEN_ROWS, (_iso(lo), _iso(settle)))
    return [_task_verify_start(moment, task, enters, exits, greens) for moment, task in enters if moment < hi]


def _first_after(moments: Sequence[TaskMoment], task: str, after: datetime) -> datetime | None:
    return min((t for t, k in moments if k == task and t > after), default=None)


def _task_verify_start(
    moment: datetime, task: str, enters: Sequence[TaskMoment], exits: Sequence[TaskMoment],
    greens: Sequence[TaskMoment],
) -> canary.TaskVerifyStart:
    next_start = _first_after(enters, task, moment)
    green = _first_after(greens, task, moment)
    exit_at = _first_after(exits, task, moment)
    if green is not None and (next_start is None or green < next_start):
        return canary.TaskVerifyStart(task_id=task, at=moment, passed=True)
    exited = exit_at is not None and (next_start is None or exit_at < next_start)
    return canary.TaskVerifyStart(task_id=task, at=moment, passed=False if exited else None)


# ---------------------------------------------------------------------------
# git: main's reflog and the gate ranges.
# ---------------------------------------------------------------------------

_REFLOG_ENTRY = re.compile(r'^(?P<sha>[0-9a-f]{40}) \S+@\{(?P<epoch>\d+)\}$')


def _git(git_root: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(git_root), *args], capture_output=True, text=True, check=True,
        timeout=GIT_TIMEOUT_SECS,
    ).stdout


def _first_parent(git_root: Path, old: str | None, new: str) -> tuple[str, ...]:
    spec = new if old is None else f'{old}..{new}'
    return tuple(_git(git_root, 'rev-list', '--first-parent', '--reverse', spec).split())


def read_main_updates(git_root: Path, since: datetime, until: datetime) -> list[canary.MainUpdate]:
    """Main's reflog updates stamped in ``[since, until)``, each with the first-parent commits it added."""
    entries = [
        (m['sha'], datetime.fromtimestamp(int(m['epoch']), UTC))
        for line in _git(git_root, 'log', '-g', '--date=unix', '--format=%H %gd', 'refs/heads/main').splitlines()
        if (m := _REFLOG_ENTRY.match(line)) is not None
    ]
    entries.reverse()
    previous = [None, *(sha for sha, _ in entries[:-1])]
    return [
        canary.MainUpdate(at=moment, new_sha=sha, commits=_first_parent(git_root, old, sha))
        for (sha, moment), old in zip(entries, previous, strict=True)
        if since <= moment < until
    ]


def read_gate_ranges(git_root: Path, gate_rows: Sequence[Event]) -> dict[tuple[str, str], tuple[str, ...]]:
    """First-parent commits of each gate's ``rebased_from..rebased_onto``; an unreadable range is left out."""
    ranges: dict[tuple[str, str], tuple[str, ...]] = {}
    for key in {k for e in gate_rows if (k := canary.rebase_range(e)) is not None}:
        try:
            ranges[key] = _first_parent(git_root, key[0], key[1])
        except subprocess.CalledProcessError:
            continue
    return ranges


# ---------------------------------------------------------------------------
# The journal extract: parsed here, at the boundary, into typed records.
# ---------------------------------------------------------------------------

_JOURNAL_LINE = re.compile(r'^(?P<at>\S+) \S+ \S+ \S+ \S+ (?P<level>[A-Z]+) +\[[^\]]+\] (?P<message>.*)$')
_SWEEP_START = re.compile(r'^Main-tip integrity sweep: verifying (?P<sha>[0-9a-f]{12}) with a (?:warm|cold) build')
_SWEEP_VERDICT = re.compile(
    r'^Main-tip integrity sweep: (?:warm|cold) build of (?P<sha>[0-9a-f]{12}) '
    r'(?P<result>passed|failed) \((?P<category>[^)]*)\)$'
)
_SWEEP_FILED = re.compile(r'^Main-tip integrity sweep: filed L1 escalation (?P<esc>\S+) for SHA (?P<sha>[0-9a-f]{12}) ')
_SWEEP_UNCONFIRMED = re.compile(r'^Main-tip integrity sweep: failure at (?P<sha>[0-9a-f]{12}) unconfirmed on current tip')
_AUDIT_FOUND = re.compile(
    r'^merge queue resource-conservation audit: \d+ violation\(s\) found \(consecutive streak=\d+\): (?P<texts>.+)$'
)
_AUDIT_REPEAT = re.compile(r'^merge queue resource-conservation audit: \d+ violation\(s\) found .*\(unchanged for ')
_AUDIT_CLEAR = re.compile(r'^merge queue resource-conservation audit clear after ')
_AUDIT_SEPARATOR = '; '


@dataclass(frozen=True)
class JournalRecords:
    sweeps: tuple[canary.SweepVerdict, ...]
    audits: tuple[canary.AuditRecord, ...]
    escalations: tuple[str, ...]
    unparsed: int


@dataclass
class _JournalAccumulator:
    started: dict[str, datetime] = field(default_factory=dict)
    sweeps: list[canary.SweepVerdict] = field(default_factory=list)
    audits: list[canary.AuditRecord] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)

    def decide(self, sha12: str, filed: bool) -> None:
        for i in range(len(self.sweeps) - 1, -1, -1):
            if self.sweeps[i].sha12 == sha12:
                self.sweeps[i] = replace(self.sweeps[i], filed=filed)
                return


Handler = Callable[[_JournalAccumulator, re.Match[str], datetime, str], None]


def _on_start(acc: _JournalAccumulator, m: re.Match[str], moment: datetime, _level: str) -> None:
    acc.started[m['sha']] = moment


def _on_verdict(acc: _JournalAccumulator, m: re.Match[str], moment: datetime, _level: str) -> None:
    acc.sweeps.append(canary.SweepVerdict(
        started_at=acc.started.pop(m['sha'], moment), verdict_at=moment, sha12=m['sha'],
        passed=m['result'] == 'passed', filed=None, category=m['category'],
    ))


def _on_filed(acc: _JournalAccumulator, m: re.Match[str], _moment: datetime, _level: str) -> None:
    acc.decide(m['sha'], filed=True)
    acc.escalations.append(m['esc'])


def _on_unconfirmed(acc: _JournalAccumulator, m: re.Match[str], _moment: datetime, _level: str) -> None:
    acc.decide(m['sha'], filed=False)


def _on_audit_found(acc: _JournalAccumulator, m: re.Match[str], moment: datetime, level: str) -> None:
    acc.audits.append(canary.AuditRecord(
        at=moment, level=canary.AuditLevel(level), violations=tuple(m['texts'].split(_AUDIT_SEPARATOR)),
    ))


def _on_audit_clear(acc: _JournalAccumulator, _m: re.Match[str], moment: datetime, _level: str) -> None:
    acc.audits.append(canary.AuditRecord(at=moment, level=canary.AuditLevel.INFO, violations=()))


def _on_repeat(_acc: _JournalAccumulator, _m: re.Match[str], _moment: datetime, _level: str) -> None:
    """A coalesced 'unchanged' repeat restates the last WARNING; it adds nothing."""


_HANDLERS: tuple[tuple[re.Pattern[str], Handler], ...] = (
    (_SWEEP_START, _on_start), (_SWEEP_VERDICT, _on_verdict), (_SWEEP_FILED, _on_filed),
    (_SWEEP_UNCONFIRMED, _on_unconfirmed), (_AUDIT_REPEAT, _on_repeat), (_AUDIT_FOUND, _on_audit_found),
    (_AUDIT_CLEAR, _on_audit_clear),
)


def _handle(acc: _JournalAccumulator, line: str) -> bool:
    parsed = _JOURNAL_LINE.match(line)
    moment = None if parsed is None else _parse_instant(parsed['at'])
    if parsed is None or moment is None or parsed['level'] not in canary.AuditLevel.__members__:
        return False
    for pattern, handler in _HANDLERS:
        if (m := pattern.match(parsed['message'])) is not None:
            handler(acc, m, moment, parsed['level'])
            return True
    return False


def parse_journal(lines: Iterable[str]) -> JournalRecords:
    """Main-tip sweep verdicts and resource-audit records from orchestrator journal lines.

    A sweep's verdict pairs with its 'verifying' line; a failure's 'filed L1'
    or 'unconfirmed' line sets ``filed`` True or False. A line of no known
    shape is counted in ``unparsed`` and never raises.
    """
    acc = _JournalAccumulator()
    unparsed = sum(1 for line in lines if line.strip() and not _handle(acc, line))
    return JournalRecords(
        sweeps=tuple(acc.sweeps), audits=tuple(acc.audits), escalations=tuple(acc.escalations),
        unparsed=unparsed,
    )


# ---------------------------------------------------------------------------
# Figures.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hosts:
    laptop: str
    local: str


@dataclass(frozen=True)
class Readout:
    lo: datetime
    hi: datetime
    sources: tuple[str, ...]
    ledger: tuple[canary.Landing, ...]
    predicates: tuple[canary.Predicate, ...]
    voids: canary.VoidTally
    conditionals: canary.GateConditionals
    local_hours: canary.HostHours
    union_busy_hours: float
    queue: canary.QueueToLand
    chain_items: dict[int, int]
    depth_reasons: dict[str, int]
    starved: canary.DepthStarved
    solos: canary.PostRedSolos
    wall: canary.ChainWall | None
    gate_split: dict[canary.GateRange, int]
    frozen: canary.FrozenTipResult
    accounting: canary.AccountingTally
    red: canary.RedEpisodes
    red_starts: canary.RedStarts
    journal: JournalRecords


@dataclass(frozen=True)
class MainHistory:
    updates: tuple[canary.MainUpdate, ...]
    gate_ranges: dict[tuple[str, str], tuple[str, ...]]


def compute(
    events: WindowEvents, task_starts: Sequence[canary.TaskVerifyStart], journal: JournalRecords,
    history: MainHistory, window: tuple[datetime, datetime], hosts: Hosts, sources: tuple[str, ...],
) -> Readout:
    lo, hi = window
    ledger = canary.landings(events.finalized, events.train_merged)
    lookback_ledger = canary.landings(events.finalized_lookback, events.train_merged_lookback)
    commits = canary.classify_commits(
        history.updates, {row.merge_sha for row in lookback_ledger if row.merge_sha is not None},
    )
    red = canary.red_episodes(journal.sweeps, commits.landed_at)
    frozen = canary.frozen_tip_violations(events.voided, events.discards, events.verify_lookback)
    accounting = canary.accounting_violations(journal.audits)
    conditionals = canary.gate_conditionals(ledger, events.gates, events.queued_lookback)
    local_hours = canary.host_verify_hours(events.verify_lookback, events.cancelled, lo, hi, host=hosts.local)
    queue = canary.queue_to_land(events.finalized)
    figures = canary.CanaryFigures(
        gates_per_landing=canary.gates_per_landing(events.gates, ledger),
        voids_per_landing=canary.voids_per_landing(events.voided, ledger).per_landing,
        p_gate_after_gate=conditionals.p_gate_after_gate,
        laptop_finished_share=canary.host_finished_share(events.heartbeats, hi, host=hosts.laptop),
        frozen_tip_violations=len(frozen.violations),
        speculation_accounting_violations=accounting.distinct_texts,
        chain_attributed_red_main=canary.chain_attributed_episodes(
            red.episodes, commits,
            [row.merge_sha for row in lookback_ledger if row.via_chain and row.merge_sha is not None],
        ),
        queue_to_land_p50_hours=queue.p50_hours,
        local_verify_hours_per_week=local_hours.per_week,
        deep_chain_p90_secs=canary.deep_chain_p90_secs(events.verify),
    )
    return Readout(
        lo=lo, hi=hi, sources=sources, ledger=ledger, predicates=canary.evaluate(figures),
        voids=canary.voids_per_landing(events.voided, ledger), conditionals=conditionals,
        local_hours=local_hours, union_busy_hours=canary.union_busy_hours(events.verify_lookback, lo, hi),
        queue=queue, chain_items=canary.chain_items_distribution(events.verify),
        depth_reasons=canary.depth_reason_tally(events.verify), starved=canary.depth_starved(events.verify),
        solos=canary.post_red_chain_solos(events.verify_lookback, events.finalized_lookback, lo, hi),
        wall=canary.per_item_chain_wall(events.verify, min_items=PER_ITEM_WALL_MIN_ITEMS),
        gate_split=canary.gate_reverify_split(events.gates, history.gate_ranges, commits.classes),
        frozen=frozen, accounting=accounting, red=red,
        red_starts=canary.verifies_started_on_red(red.episodes, events.verify, events.cancelled, task_starts),
        journal=journal,
    )


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------


def _num(value: float | None) -> str:
    return 'not measured' if value is None else canary.render_number(value)


def _pct(value: float | None) -> str:
    return 'not measured' if value is None else f'{value:.1%}'


def _when(moment: datetime | None) -> str:
    return 'open' if moment is None else moment.isoformat(timespec='seconds')


def _cell(text: str) -> str:
    return text.replace('|', '\\|')


def _predicate_table(rows: Sequence[canary.Predicate]) -> list[str]:
    lines = [
        '## Predicates (plans/merge-lane-policy-prd.md D12)', '',
        '| predicate | value | limit | verdict | rule | baseline |', '|---|---|---|---|---|---|',
    ]
    lines += [
        f'| {_cell(r.label)} | {r.rendered_value} | {r.rendered_limit} | {"holds" if r.holds else "FALSE"} '
        f'| {_cell(r.rule)} | {r.baseline} |'
        for r in rows
    ]
    return lines


def _landings_section(r: Readout) -> list[str]:
    return [
        '## Landings and lane hours', '',
        f'- landings: {len(r.ledger)} (merge_finalized done ∪ train_merged members); '
        f'landed via a chain: {sum(row.via_chain for row in r.ledger)}',
        f'- local merge-verify hours: {_num(r.local_hours.hours)} h in the window, '
        f'{_num(r.local_hours.per_week)} h/week (union of completed and cancelled intervals)',
        f'- union-busy hours, all hosts: {_num(r.union_busy_hours)} h (union of completed merge-verify intervals)',
        f'- median queue-to-land: {_num(r.queue.p50_hours)} h over {r.queue.n} landings '
        f'({r.queue.n_missing} without queue_to_land_secs)',
    ]


def _voids_section(r: Readout) -> list[str]:
    c = r.conditionals
    return [
        '## Voids', '',
        f'- verdict_voided rows: {r.voids.total}; before a host (point dispatch): {r.voids.before_host}; '
        f'at adoption: {r.voids.at_adoption}; per landing: {_num(r.voids.per_landing)}',
        f'- gated landings: {sum(c.gated)} of {len(c.gated)}; P(gate after a gated landing): '
        f'{_num(c.p_gate_after_gate)}; P(gate after an ungated landing): {_num(c.p_gate_after_ungated)}',
    ]


def _chain_sections(r: Readout) -> list[str]:
    s = r.starved
    wall = (
        'no chain_items >= 8 row in the window' if r.wall is None
        else f'n={r.wall.n}, p50 {_num(r.wall.p50_secs)} s/item, max {_num(r.wall.max_secs)} s/item'
    )
    return [
        '## chain_items distribution', '',
        *[f'- {items}: {n}' for items, n in sorted(r.chain_items.items())], '',
        '## Depth-reason tally', '',
        *[f'- {basis}: {n}' for basis, n in sorted(r.depth_reasons.items())], '',
        '## Depth-starved verifies', '',
        f'- policy-sized: {s.sized}; starved (chain_items < the policy row depth): {s.starved} '
        f'(too few waiting: {s.too_few_waiting}, other: {s.other}); share of merge verifies: '
        f'{_pct(s.share_of_verifies)}', '',
        '## Post-red-chain solos', '',
        f'- solo verifies of an unretired red-chain member: {r.solos.count}; {_num(r.solos.hours)} host-hours, '
        f'{_pct(r.solos.share_of_verify_hours)} of completed merge-verify hours; next finalize blocked: '
        f'{r.solos.next_finalize_blocked}', '',
        f'## Per-item chain wall (chain_items >= {PER_ITEM_WALL_MIN_ITEMS})', '',
        f'- {wall}',
    ]


def _gate_section(r: Readout) -> list[str]:
    return [
        '## Gate re-verify split', '',
        'Each reverify=true gate by the first-parent commits of rebased_from..rebased_onto, '
        "classed by main's reflog: a commit an update landing a done merge_sha added is queue, "
        'every other one direct.', '',
        *[f'- {kind}: {n}' for kind, n in r.gate_split.items()],
    ]


def _invariant_sections(r: Readout) -> list[str]:
    a = r.accounting
    return [
        '## Frozen-tip anchors', '',
        f'- live_tip anchors examined: {r.frozen.anchors}; onto a tip recorded dead earlier: '
        f'{len(r.frozen.violations)}; tip recorded dead at or after the anchor: {r.frozen.died_after_anchor}',
        *[f'  - task {task} at {_when(moment)} onto {base}' for task, moment, base in r.frozen.violations], '',
        '## Speculation accounting (orchestrator journal)', '',
        f'- distinct speculation-accounting violation texts: {a.distinct_texts}; audit records carrying one: '
        f'{a.records_with_any}; worktree-ledger (I6) records: {a.worktree_ledger_records}',
    ]


def _episode_line(ep: canary.RedEpisode) -> str:
    return (
        f'- first red {ep.first_red_sha12} (opened by the filed failure at {ep.opening_sha12}), '
        f'green {ep.green_sha12 or "none"}; span [{_when(ep.start)}, {_when(ep.end)}); candidate range after '
        f'{_when(ep.candidate_after)}; {ep.red_verdicts} red verdicts'
    )


def _started_line(kind: str, s: canary.StartedOnRed) -> str:
    return f'- {kind} verifies started on red: {s.started} of {s.total} ({_pct(s.share)}), {s.failed} failed'


def _red_section(r: Readout) -> list[str]:
    unconfirmed = [v for v in r.journal.sweeps if not v.passed and v.filed is False]
    return [
        '## Red-main episodes', '',
        *(_episode_line(ep) for ep in r.red.episodes),
        f'- episodes: {len(r.red.episodes)}; unconfirmed sweep failures: {r.red.unconfirmed}',
        *[f'  - {v.sha12} ({v.category}) at {_when(v.verdict_at)}' for v in unconfirmed],
        _started_line('merge', r.red_starts.merge),
        _started_line('task', r.red_starts.task),
        f'- L1 escalations filed by the sweep: {", ".join(r.journal.escalations) or "none"}',
    ]


def _false_section(r: Readout) -> list[str]:
    false_rows = canary.false_predicates(r.predicates)
    return [
        '## False predicates', '',
        *[f'- {row.label}: {row.rendered_value} (limit {row.rendered_limit})' for row in false_rows],
        *(['- none'] if not false_rows else []),
        f'- recommends merge_deep.chain_cap: 0: {"yes" if canary.recommends_chain_cap_zero(r.predicates) else "no"}',
    ]


def render(r: Readout) -> str:
    header = [
        f'# Merge-lane canary readout: [{_when(r.lo)}, {_when(r.hi)})', '',
        'Generated by scripts/merge_lane_canary_readout.py from:',
        *[f'- {source}' for source in r.sources],
        f'- journal extract: {len(r.journal.sweeps)} sweep verdicts, {len(r.journal.audits)} audit records, '
        f'{r.journal.unparsed} unparsed lines',
    ]
    sections = (
        header, _predicate_table(r.predicates), _landings_section(r), _voids_section(r), _chain_sections(r),
        _gate_section(r), _invariant_sections(r), _red_section(r), _false_section(r),
    )
    return '\n\n'.join('\n'.join(lines) for lines in sections) + '\n'


def render_refusal(missing: Sequence[str], lo: datetime, hi: datetime) -> str:
    return '\n'.join([
        "REFUSED: the window's rows predate the merge-lane policy's D10 alpha fields.", '',
        f'Window: [{_when(lo)}, {_when(hi)})', '',
        'No row in the window carries:',
        *[f'- {name}' for name in missing], '',
        'No predicate is evaluated: a canary must not pass on missing signal.',
    ]) + '\n'


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument('--window', required=True, help='<iso>..<iso>, half-open')
    parser.add_argument('--project-root', help="the project whose runs.db is read (default: this checkout's main)")
    parser.add_argument('--git-root', help="the checkout whose main reflog is read (default: the project root)")
    parser.add_argument('--journal-extract', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--laptop-host', default='laptop')
    parser.add_argument('--local-host', default='local')
    return parser


def _fault(message: object) -> int:
    print(message, file=sys.stderr)
    return EXIT_FAULT


def main(argv: Sequence[str], now: datetime | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lo, hi = mlt.parse_window(args.window, now or datetime.now(UTC))
    except argparse.ArgumentTypeError as exc:
        return _fault(exc)
    root = mlt.resolve_project_roots([args.project_root] if args.project_root else None)[0]
    git_root = Path(args.git_root).resolve() if args.git_root else root
    db = runs_db(root)
    try:
        with closing(_connect_ro(db)) as conn:
            events = load_window(conn, lo, hi)
            task_starts = load_task_verify_starts(conn, lo, hi)
    except sqlite3.Error as exc:
        return _fault(f'runs.db read failed: {db}: {exc}')
    missing = canary.missing_alpha_fields(events.verify, events.cancelled, events.voided, events.finalized)
    if missing:
        args.out.write_text(render_refusal(missing, lo, hi), encoding='utf-8')
        return EXIT_REFUSED
    try:
        journal = parse_journal(args.journal_extract.read_text(encoding='utf-8').splitlines())
    except (OSError, UnicodeDecodeError) as exc:
        return _fault(f'journal extract read failed: {exc}')
    try:
        history = MainHistory(
            updates=tuple(read_main_updates(git_root, lo - LOOKBACK, hi)),
            gate_ranges=read_gate_ranges(git_root, events.gates),
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return _fault(f'git read failed: {exc}')
    sources = (f'runs.db: {db}', f"main's reflog: {git_root}", f'journal extract: {args.journal_extract}')
    readout = compute(events, task_starts, journal, history, (lo, hi), Hosts(args.laptop_host, args.local_host), sources)
    args.out.write_text(render(readout), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
