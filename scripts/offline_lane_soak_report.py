#!/usr/bin/env python3
"""The offline-lane soak report: ζ's mechanical checks and its re-measurement figures (task 6867).

PRD ``plans/offline-lane-red-path-prd.md`` §9 ζ and ζ0.  Read-only.  Reads the lane's
run records, its episode records, the flake ledger, the escalation queue and the task
store through their owners (``offline_lane_soak_inputs.py``), then judges them here.

Exit codes, the contract ζ's runbook reads:
    0  every check holds (a check that examined nothing reports no_data, and passes the run)
    1  any check fails
    2  nothing to judge: a store is missing or unreadable, or no pass record is in the window

Usage:
    python scripts/offline_lane_soak_report.py [--project-root ROOT] [--since ISO]
        [--command NAME ...] [--json] [--server-url URL] [--recheck-secs N]

Verdict lines go to stderr; stdout carries the figures, or with --json the full report.
Run it directly, never via `python -m`: it imports flat siblings in scripts/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import _checkout as _checkout
from migrate_metadata_modules_to_files import DEFAULT_SERVER
from offline_lane_soak_inputs import SoakInputs, TaskReader, Unjudgeable, WindowPass, load_inputs
from orchestrator.flake_ledger import (
    DUE_ITEM_STALE_AFTER,
    DUE_ITEM_STALE_MINUTES,
    UNKNOWN_TEST_ID,
    FlakeCallSite,
    FlakeVerdict,
)
from orchestrator.offline_lane_confirm import Confirmed
from orchestrator.offline_lane_episode_state import LanePolicy, OfferedDebt, unconfirmable_node
from orchestrator.offline_lane_filing import LANE_METADATA_KEY
from orchestrator.offline_lane_records import FailureRecord, PassRecord
from shared.project_layout import default_project_root

# The de-flake owner's metadata keys, by their PRD section 5.6 names: flake_ledger spells them
# inline (orchestrator/src/orchestrator/flake_ledger.py::_deflake_metadata) with no public constant.
DEBT_CALL_SITE_KEY = 'flake_debt_call_site'
DEBT_TRIGGER_KEY = 'flake_debt_trigger'


class Verdict(StrEnum):
    PASS = 'pass'
    FAIL = 'fail'
    NO_DATA = 'no_data'


class Check(StrEnum):
    RECORDS_WELL_FORMED = 'records_well_formed'
    FILINGS_RESOLVE = 'filings_resolve'
    FILINGS_BOUNDED_BY_THRESHOLD = 'filings_bounded_by_threshold'
    LEDGER_ROWS_PER_PASS = 'ledger_rows_per_pass'
    DEFLAKE_TASKS_CARRY_EVIDENCE = 'deflake_tasks_carry_evidence'
    NO_STRANDED_FILING = 'no_stranded_filing'
    OFFERED_DEBTS_OWNED_OR_DUE = 'offered_debts_owned_or_due'


@dataclass(frozen=True)
class CheckResult:
    name: Check
    verdict: Verdict
    examined: int
    rule: str
    failures: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Figure:
    name: str
    rule: str
    value: Any


@dataclass(frozen=True)
class Settings:
    recheck_secs: float


def _rules(settings: Settings) -> dict[Check, str]:
    return {
        Check.RECORDS_WELL_FORMED: 'every pass directory in the window holds a record.json, outcome block '
        'and effect block their owners parse, and each finished sub-run output file exists; a pass with no '
        'detect record, or with sub-runs and no effect block, is interrupted and does not fail',
        Check.FILINGS_RESOLVE: 'every task id a windowed effect block lists in tasks_filed resolves through '
        f'get_task with a non-empty details digest and an object at metadata.{LANE_METADATA_KEY} (PRD 5.5)',
        Check.FILINGS_BOUNDED_BY_THRESHOLD: 'the distinct task ids the windowed effect blocks list in tasks_filed '
        'number no more than the (pass, test) entries they list in threshold_reached',
        Check.LEDGER_ROWS_PER_PASS: 'every test a windowed confirmed outcome lists as reproduced has a '
        'fails_in_isolation row, and every cleared test a passes_in_isolation row, at call site offline_lane '
        "observed at the pass's start (PRD 5.8); the unconfirmable pseudo-test and the unknown id are exempt",
        Check.DEFLAKE_TASKS_CARRY_EVIDENCE: 'every owner task of a debt a windowed effect block offered, whose '
        'metadata says flake_debt_call_site offline_lane, carries a flake_debt_trigger and, in its description '
        "or details, the kind and first line of one of that test's failures in the command's pass records",
        Check.NO_STRANDED_FILING: 'every test a windowed effect block lists in threshold_reached is covered by a '
        'lane task (filed in a windowed effect block, or bound to its live episode), or its filing is due and '
        "the command's stuck-filing record is pending; a filing due for less than "
        f'{settings.recheck_secs:g}s is pending, not stranded',
        Check.OFFERED_DEBTS_OWNED_OR_DUE: 'every test a windowed effect block lists in debts_offered has a debt '
        'row with an owner task, or is still on the due list; due items older than '
        f'{DUE_ITEM_STALE_MINUTES} minutes are reported',
    }


def _judged(
    check: Check, rule: str, examined: int, failures: Sequence[str] = (), notes: Sequence[str] = (),
) -> CheckResult:
    return CheckResult(check, _verdict(examined, failures), examined, rule, tuple(failures), tuple(notes))


def _verdict(examined: int, failures: Sequence[str]) -> Verdict:
    if examined == 0:
        return Verdict.NO_DATA
    return Verdict.FAIL if failures else Verdict.PASS


CheckFn = Callable[[SoakInputs, str, Settings], CheckResult]


def _not_yet_judged(check: Check) -> CheckFn:
    def judge(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
        return _judged(check, rule, 0)
    return judge


def _label(window_pass: WindowPass) -> str:
    return f'{window_pass.stored.command_key}/{window_pass.pass_id}'


def _interrupted(window_pass: WindowPass) -> bool:
    """A header with no sub-run, or sub-runs and no effect block: a restart cut the pass short."""
    record = window_pass.stored.record
    return record is not None and (record.detect is None or record.effect is None)


def _records_well_formed(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
    failures = [f'{_label(p)}: {problem}' for p in inputs.passes for problem in p.problems]
    failures += [
        f'{_label(p)}: the {phase} sub-run output file is missing' for p in inputs.passes for phase in p.missing_outputs
    ]
    notes = [f'interrupted: {_label(p)}' for p in inputs.passes if _interrupted(p)]
    return _judged(Check.RECORDS_WELL_FORMED, rule, len(inputs.passes), failures, notes)


def _filing_problem(task: Mapping[str, Any] | None) -> str | None:
    if task is None:
        return 'get_task found no such task'
    details = task.get('details')
    if not isinstance(details, str) or not details.strip():
        return 'its details digest is empty'
    metadata = task.get('metadata')
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get(LANE_METADATA_KEY), Mapping):
        return f'it carries no metadata.{LANE_METADATA_KEY}'
    return None


def _filings_resolve(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
    filed_by: dict[str, str] = {}
    for window_pass, effect in inputs.effects():
        for task_id in effect.tasks_filed:
            filed_by.setdefault(task_id, _label(window_pass))
    failures = [
        f'task {task_id} (filed by {label}): {problem}'
        for task_id, label in filed_by.items()
        if (problem := _filing_problem(inputs.tasks.get(task_id))) is not None
    ]
    return _judged(Check.FILINGS_RESOLVE, rule, len(filed_by), failures)


def _filings_bounded_by_threshold(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
    blocks = [effect for _, effect in inputs.effects() if effect.tasks_filed or effect.threshold_reached]
    filed = len({task_id for effect in blocks for task_id in effect.tasks_filed})
    crossings = sum(len(set(effect.threshold_reached)) for effect in blocks)
    counts = f'{filed} distinct task(s) filed; {crossings} (pass, test) threshold crossing(s)'
    if filed > crossings:
        return _judged(Check.FILINGS_BOUNDED_BY_THRESHOLD, rule, len(blocks), failures=[counts])
    return _judged(Check.FILINGS_BOUNDED_BY_THRESHOLD, rule, len(blocks), notes=[counts])


def _lane_rows(inputs: SoakInputs) -> set[tuple[str, datetime, str]]:
    return {
        (row.test_id, datetime.fromisoformat(row.observed_at), row.verdict)
        for row in inputs.ledger.occurrences
        if row.call_site == FlakeCallSite.offline_lane
    }


def _owed_rows(record: PassRecord, outcome: Confirmed) -> list[tuple[str, FlakeVerdict]]:
    exempt = {unconfirmable_node(record.command), UNKNOWN_TEST_ID}
    owed = [(node, FlakeVerdict.fails_in_isolation) for node in outcome.reproduced]
    owed += [(node, FlakeVerdict.passes_in_isolation) for node in outcome.cleared]
    return [(node, verdict) for node, verdict in owed if node not in exempt]


def _ledger_rows_per_pass(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
    rows = _lane_rows(inputs)
    examined = 0
    failures: list[str] = []
    for window_pass in inputs.passes:
        record = window_pass.stored.record
        if record is None or not isinstance(window_pass.outcome, Confirmed):
            continue
        owed = _owed_rows(record, window_pass.outcome)
        examined += len(owed)
        missing = [f'{node} ({verdict})' for node, verdict in owed if (node, record.started_at, verdict) not in rows]
        if missing:
            failures.append(f'{_label(window_pass)}: no offline_lane row for ' + ', '.join(missing))
    return _judged(Check.LEDGER_ROWS_PER_PASS, rule, examined, failures)


def _offered(inputs: SoakInputs) -> Iterator[tuple[WindowPass, PassRecord, OfferedDebt]]:
    for window_pass, effect in inputs.effects():
        record = window_pass.stored.record
        if record is not None:
            for debt in effect.debts_offered:
                yield window_pass, record, debt


def _metadata(task: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = task.get('metadata')
    return metadata if isinstance(metadata, Mapping) else {}


def _text(task: Mapping[str, Any]) -> str:
    return '\n'.join(value for key in ('description', 'details') if isinstance(value := task.get(key), str))


def _evidence_problem(owner: Mapping[str, Any], failures: Sequence[FailureRecord]) -> str | None:
    trigger = _metadata(owner).get(DEBT_TRIGGER_KEY)
    if not isinstance(trigger, str) or not trigger:
        return f'it carries no metadata.{DEBT_TRIGGER_KEY}'
    text = _text(owner)
    if not any(failure.kind in text and failure.first_line in text for failure in failures):
        return "its text carries no failure kind and first line of the test from the lane's pass records"
    return None


def _deflake_tasks_carry_evidence(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
    examined: dict[str, tuple[str, str]] = {}
    for _, record, debt in _offered(inputs):
        row = inputs.ledger.debts.get(debt.node)
        owner = None if row is None or row.owner_task_id is None else inputs.tasks.get(row.owner_task_id)
        if row is None or owner is None or _metadata(owner).get(DEBT_CALL_SITE_KEY) != FlakeCallSite.offline_lane:
            continue
        examined.setdefault(str(row.owner_task_id), (record.command, debt.node))
    failures = [
        f'owner task {owner_id} of {node}: {problem}'
        for owner_id, (command, node) in examined.items()
        if (owner := inputs.tasks.get(owner_id)) is not None
        and (problem := _evidence_problem(owner, inputs.failures.get((command, node), ()))) is not None
    ]
    return _judged(Check.DEFLAKE_TASKS_CARRY_EVIDENCE, rule, len(examined), failures)


def _stale_notes(inputs: SoakInputs) -> list[str]:
    stale_before = inputs.now - DUE_ITEM_STALE_AFTER
    return [
        f'stale due item: {item.test_id} at {item.call_site}, enqueued {item.enqueued_at}, '
        f'{item.attempts} attempt(s), last error: {item.last_error or "none"}'
        for item in inputs.ledger.due_items
        if datetime.fromisoformat(item.enqueued_at) <= stale_before
    ]


def _offered_debts_owned_or_due(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
    pending = inputs.ledger.due_summary.pending_since
    offered = list(_offered(inputs))
    failures = [
        f'{_label(window_pass)}: {debt.node} ({debt.trigger.value}) has no debt owner and is not on the due list'
        for window_pass, _, debt in offered
        if debt.node not in pending
        and ((row := inputs.ledger.debts.get(debt.node)) is None or row.owner_task_id is None)
    ]
    return _judged(Check.OFFERED_DEBTS_OWNED_OR_DUE, rule, len(offered), failures, _stale_notes(inputs))


_CHECKS: dict[Check, CheckFn] = {
    **{check: _not_yet_judged(check) for check in Check},
    Check.RECORDS_WELL_FORMED: _records_well_formed,
    Check.FILINGS_RESOLVE: _filings_resolve,
    Check.FILINGS_BOUNDED_BY_THRESHOLD: _filings_bounded_by_threshold,
    Check.LEDGER_ROWS_PER_PASS: _ledger_rows_per_pass,
    Check.DEFLAKE_TASKS_CARRY_EVIDENCE: _deflake_tasks_carry_evidence,
    Check.OFFERED_DEBTS_OWNED_OR_DUE: _offered_debts_owned_or_due,
}


def run_checks(inputs: SoakInputs | Unjudgeable, settings: Settings) -> tuple[CheckResult, ...]:
    rules = _rules(settings)
    if isinstance(inputs, Unjudgeable):
        return tuple(_judged(check, rules[check], 0) for check in Check)
    return tuple(_CHECKS[check](inputs, rules[check], settings) for check in Check)


def _passes_per_command_per_day(inputs: SoakInputs) -> Figure:
    days: dict[str, Counter[str]] = defaultdict(Counter)
    for p in inputs.passes:
        days[p.stored.command_key][p.started_at.date().isoformat()] += 1
    return Figure(
        'passes_per_command_per_day',
        'pass directories in the window by command key and the UTC date of their pass-id start; '
        "completeness against an independent count is the resolver's judgment",
        {command: dict(sorted(count.items())) for command, count in sorted(days.items())},
    )


def _interrupted_passes(inputs: SoakInputs) -> Figure:
    return Figure(
        'interrupted_passes',
        'passes in the window with a header and no detect record, or sub-runs and no effect block',
        [{'command': p.stored.command_key, 'pass_id': p.pass_id} for p in inputs.passes if _interrupted(p)],
    )


_FIGURES: tuple[Callable[[SoakInputs], Figure], ...] = (
    _passes_per_command_per_day,
    _interrupted_passes,
)


def figures(inputs: SoakInputs | Unjudgeable) -> tuple[Figure, ...]:
    if isinstance(inputs, Unjudgeable):
        return ()
    return tuple(figure(inputs) for figure in _FIGURES)


def exit_code(inputs: SoakInputs | Unjudgeable, results: Sequence[CheckResult]) -> int:
    if isinstance(inputs, Unjudgeable):
        return 2
    return 1 if any(result.verdict is Verdict.FAIL for result in results) else 0


def verdict_lines(results: Sequence[CheckResult]) -> list[str]:
    lines: list[str] = []
    for result in results:
        lines.append(f'[{result.verdict.upper()}] {result.name}: examined {result.examined}; rule: {result.rule}')
        lines.extend(f'  - {failure}' for failure in result.failures)
    return lines


def _report_json(
    inputs: SoakInputs | Unjudgeable, now: datetime, results: Sequence[CheckResult], shown: Sequence[Figure],
) -> dict[str, Any]:
    judged = not isinstance(inputs, Unjudgeable)
    return {
        'exit_code': exit_code(inputs, results),
        'unjudgeable': None if judged else inputs.reason,
        'since': inputs.since.isoformat() if judged else None,
        'now': now.isoformat(),
        'commands': list(inputs.commands) if judged else [],
        'checks': [
            {
                'name': r.name.value, 'verdict': r.verdict.value, 'examined': r.examined, 'rule': r.rule,
                'failures': list(r.failures), 'notes': list(r.notes),
            }
            for r in results
        ],
        'figures': [{'name': f.name, 'rule': f.rule, 'value': f.value} for f in shown],
    }


def render_figures(shown: Sequence[Figure]) -> str:
    return ''.join(f'{f.name} ({f.rule}):\n  {json.dumps(f.value, sort_keys=True)}\n' for f in shown)


def _instant(text: str) -> datetime:
    try:
        instant = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'not an ISO-8601 instant: {text!r}') from exc
    if instant.tzinfo is None:
        raise argparse.ArgumentTypeError(f'--since needs a UTC offset: {text!r}')
    return instant


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The offline-lane soak report: zeta's checks and figures.")
    parser.add_argument('--project-root', type=Path, default=None,
                        help='project whose lane to judge (default: the main checkout)')
    parser.add_argument('--since', type=_instant, default=None,
                        help='window start, ISO-8601 with an offset (default: the first effect-carrying pass)')
    parser.add_argument('--command', action='append', default=[], dest='commands',
                        help='a lane command name to judge; repeatable (default: every command with records)')
    parser.add_argument('--json', action='store_true', help='print the full report as JSON on stdout')
    parser.add_argument('--server-url', default=DEFAULT_SERVER, help='fused-memory MCP server for task reads')
    parser.add_argument('--recheck-secs', type=float, default=LanePolicy().recheck_secs,
                        help="how long a due filing may stay due before it is stranded (default: the lane's)")
    return parser


def main(
    argv: Sequence[str] | None = None, *, task_reader: TaskReader | None = None, now: datetime | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if task_reader is None:
        parser.error('no task reader')
    now = now or datetime.now(UTC)
    project_root: Path = args.project_root or default_project_root()
    settings = Settings(recheck_secs=args.recheck_secs)
    inputs = asyncio.run(load_inputs(
        project_root, since=args.since, commands=args.commands, now=now, task_reader=task_reader,
    ))
    results = run_checks(inputs, settings)
    shown = figures(inputs)
    if isinstance(inputs, Unjudgeable):
        print(f'offline-lane soak: nothing to judge: {inputs.reason}', file=sys.stderr)
    print('\n'.join(verdict_lines(results)), file=sys.stderr)
    if args.json:
        print(json.dumps(_report_json(inputs, now, results, shown), indent=2, sort_keys=True))
    else:
        print(render_figures(shown), end='')
    return exit_code(inputs, results)


if __name__ == '__main__':
    sys.exit(main())
