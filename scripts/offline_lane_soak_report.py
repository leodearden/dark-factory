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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import _checkout as _checkout
from migrate_metadata_modules_to_files import DEFAULT_SERVER
from offline_lane_soak_inputs import SoakInputs, TaskReader, Unjudgeable, load_inputs
from orchestrator.flake_ledger import DUE_ITEM_STALE_MINUTES
from orchestrator.offline_lane_episode_state import LanePolicy
from orchestrator.offline_lane_filing import LANE_METADATA_KEY
from shared.project_layout import default_project_root


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


_CHECKS: dict[Check, CheckFn] = {check: _not_yet_judged(check) for check in Check}


def run_checks(inputs: SoakInputs | Unjudgeable, settings: Settings) -> tuple[CheckResult, ...]:
    rules = _rules(settings)
    if isinstance(inputs, Unjudgeable):
        return tuple(_judged(check, rules[check], 0) for check in Check)
    return tuple(_CHECKS[check](inputs, rules[check], settings) for check in Check)


def figures(inputs: SoakInputs | Unjudgeable) -> tuple[Figure, ...]:
    if isinstance(inputs, Unjudgeable):
        return ()
    return ()


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
