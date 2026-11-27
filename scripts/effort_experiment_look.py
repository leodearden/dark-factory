#!/usr/bin/env python3
"""The effort experiment's weekly look: a predicate over the landed-per-dollar ratio R.

Each look reads the cohort of the k whole weeks since the experiment started,
k = floor((now - lag - since) / 7 days), where since is the first
routing_decision whose rule id carries --rule-prefix. The 7-day lag is there
because 85% of landings came within 7 days of a task's first implementer
dispatch, so a younger cohort week has not had its chance to land.

Why each look reads R's interval at 97%: the look runs weekly, seven looks from
week 2 to week 8, and stops at the first whose interval excludes 1. Stopping at
the first of several looks calls a direction more often than any single look
does, so each look must be stricter than the whole. Seven looks at 97% hold a
false direction call at 10% in total, 5% each way; a plain 90% interval checked
weekly calls a direction 27% of the time. At 97% a true 20% gain is found 89% of
the time, after 4.4 cohort weeks on average, and a 15% gain 69% of the time.
Source: the dark-factory ledger simulation of 2026-11-17 (tasks first
dispatched 2026-09-28 to 2026-11-08, 20,000 random splits, 185 tasks a week).

Every option the look does not define passes through to
``routing_experiment_readout.py --throughput`` (see ``--throughput --help``),
which computes R. --cohort-days and --read-at are set by the schedule: the
cohort is 7 * min(k, last look week) days, read at now.

Exit codes (docs/task-authoring.md section 6):
  0   a verdict, printed as VERDICT gain (R's interval wholly above 1),
      VERDICT loss (wholly below 1) or VERDICT no-difference-at-cap (1 inside
      the interval at or past the last look week)
  75  not yet: before the first look week, or undecided before the cap
  1   anything else: a COMPLIANCE FAILURE, no prefixed routing_decision, an
      unreadable store or history, an undefined interval, or a bad option
Once since is resolved, stdout ends with one JSON object, the look's verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import NoReturn

import _checkout as _checkout
from routing_experiment_readout import throughput_options
from routing_throughput import (
    IntervalSide,
    ThroughputReport,
    ThroughputUnavailable,
    render_text,
    resolve_since,
    summary_record,
    throughput_report,
)

LOOK_WEEK = timedelta(days=7)


class LookVerdict(Enum):
    GAIN = 'gain'
    LOSS = 'loss'
    NO_DIFFERENCE_AT_CAP = 'no-difference-at-cap'
    UNDECIDED = 'undecided'
    NOT_DUE = 'not-due'
    COMPLIANCE_FAILURE = 'compliance-failure'
    ERROR = 'error'


_EXIT = {
    LookVerdict.GAIN: 0,
    LookVerdict.LOSS: 0,
    LookVerdict.NO_DIFFERENCE_AT_CAP: 0,
    LookVerdict.UNDECIDED: 75,
    LookVerdict.NOT_DUE: 75,
    LookVerdict.COMPLIANCE_FAILURE: 1,
    LookVerdict.ERROR: 1,
}
_DECIDED = frozenset(verdict for verdict, status in _EXIT.items() if status == 0)


class _LookParser(argparse.ArgumentParser):
    """A parser whose usage error exits 1: a predicate waits only by exiting 75."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(_EXIT[LookVerdict.ERROR], f'{self.prog}: error: {message}\n')


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f'must be >= 0, got {value}')
    return value


def _look_week(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'must be >= 1, a cohort of at least one week, got {value}')
    return value


def _parser() -> _LookParser:
    parser = _LookParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument('--salt', required=True, help="the experiment's split salt")
    parser.add_argument('--rule-prefix', required=True, help='the prefix of every treated rule id')
    parser.add_argument('--lag-days', type=_non_negative_int, default=7)
    parser.add_argument('--first-look-week', type=_look_week, default=2)
    parser.add_argument('--last-look-week', type=_look_week, default=8)
    parser.add_argument('--interval-level', type=float, default=0.97)
    return parser


def _warn(message: str) -> None:
    print(f'effort_experiment_look: {message}', file=sys.stderr)


def _verdict(report: ThroughputReport, look_week: int, last_look_week: int) -> LookVerdict:
    if report.compliance.failures:
        return LookVerdict.COMPLIANCE_FAILURE
    side = report.interval_side
    if side is IntervalSide.ABOVE_1:
        return LookVerdict.GAIN
    if side is IntervalSide.BELOW_1:
        return LookVerdict.LOSS
    if side is IntervalSide.UNDEFINED:
        _warn("R's interval is undefined: no control landing, or an arm costs nothing")
        return LookVerdict.ERROR
    return LookVerdict.NO_DIFFERENCE_AT_CAP if look_week >= last_look_week else LookVerdict.UNDECIDED


def _run_look(args: argparse.Namespace, rest: Sequence[str], now: datetime) -> int:
    try:
        options = throughput_options([
            *rest, '--salt', args.salt, '--rule-prefix', args.rule_prefix,
            '--interval-level', str(args.interval_level), '--read-at', now.isoformat(),
        ])
    except SystemExit as exc:
        if exc.code:
            return _EXIT[LookVerdict.ERROR]
        raise
    since = resolve_since(options)
    look_week = (now - timedelta(days=args.lag_days) - since) // LOOK_WEEK
    if look_week < args.first_look_week:
        print(json.dumps({
            'salt': options.salt, 'since': since.astimezone(UTC).isoformat(),
            'look_week': look_week, 'verdict': LookVerdict.NOT_DUE.value,
        }))
        return _EXIT[LookVerdict.NOT_DUE]
    cohort_days = 7 * min(look_week, args.last_look_week)
    report = throughput_report(replace(options, since=since, cohort_days=cohort_days))
    sys.stdout.write(render_text(report))
    verdict = _verdict(report, look_week, args.last_look_week)
    if verdict in _DECIDED:
        print(f'VERDICT {verdict.value}')
    print(json.dumps({**summary_record(report), 'look_week': look_week, 'verdict': verdict.value}))
    return _EXIT[verdict]


def main(
    argv: Sequence[str] | None = None,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    parser = _parser()
    args, rest = parser.parse_known_args(argv)
    if args.first_look_week > args.last_look_week:
        parser.error(
            f'--first-look-week {args.first_look_week} is after --last-look-week '
            f'{args.last_look_week}'
        )
    try:
        return _run_look(args, rest, clock())
    except ThroughputUnavailable as exc:
        _warn(str(exc))
    except Exception as exc:
        _warn(f'{type(exc).__name__}: {exc}')
    return _EXIT[LookVerdict.ERROR]


if __name__ == '__main__':
    sys.exit(main())
