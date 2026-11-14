"""The live payload predicate: every budgeted request, measured against the running dashboard.

This is the before_done predicate of the ``dashboard-payload-budgets``
recurrence chain, which task 5835 seeds and which runs it daily through
``scripts/check_dashboard_payload_budgets.sh``. The pre-merge sweep
(``dashboard/tests/test_payload_budget_sweep.py``) measures a synthetic fleet;
this measures the real one, so it is what sees ORGANIC growth.

Exit codes. The deterministic runner reads only the code:

* 0: every request is within its budget, and the link is done.
* 1: at least one request is over budget. The link files a born-at-L2
  ``milestone_check_failed`` whose detail carries this output.
* 2: the instrument failed and nothing measured over. The dashboard was
  unreachable, a request timed out or answered non-200, or the table could
  not be read.

An instrument failure is never reported as an overage. A measured overage
stays exit 1 even beside a failure, because it is a verdict either way.

Output, in order: one line per request in table order; then each overage's
largest leaf fields, worst last; then ONE compact JSON summary line of at
most :data:`SUMMARY_MAX_CHARS`. The runner merges stdout and stderr and keeps
only their last 2,000 characters as the escalation detail, and it keeps a
trailing JSON line as the done note, so the lines that must survive come
last. Requests run one at a time: a measurement must not compete with the
next one for the dashboard.

Each request has a wall-clock deadline, checked as each chunk of its body
arrives, so a dashboard trickling a large body cannot hold the run. Every
network operation is also capped by the same timeout, so a request ends
within twice the timeout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from dashboard.payload_budgets import (
    TABLE_PATH,
    BudgetTable,
    EndpointBudget,
    PayloadBudgetTableError,
    field_breakdown,
    load_budget_table,
    render_overage,
)

DEFAULT_BASE_URL = 'http://127.0.0.1:8080'
DEFAULT_TIMEOUT_SECONDS = 60.0

SUMMARY_MAX_CHARS = 400
"""The deterministic runner's cap on a predicate's done note
(orchestrator/src/orchestrator/predicate_output.py::PREDICATE_NOTE_MAX_PAYLOAD_CHARS)."""


@dataclass(frozen=True)
class CheckReport:
    """The verdict and the lines that explain it, summary last."""

    exit_code: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _Measured:
    entry: EndpointBudget
    size: int

    @property
    def ratio(self) -> float:
        return self.size / self.entry.budget_bytes

    def line(self) -> str:
        return (
            f'{self.entry.url_suffix()}: {self.size:,} bytes of a '
            f'{self.entry.budget_bytes:,}-byte budget ({self.ratio:.2f})'
        )


@dataclass(frozen=True)
class _Over(_Measured):
    explanation: str


@dataclass(frozen=True)
class _Unmeasured:
    entry: EndpointBudget
    cause: str

    def line(self) -> str:
        return f'{self.entry.url_suffix()}: NOT MEASURED, {self.cause}'


def _judged(entry: EndpointBudget, body: bytes) -> _Measured:
    """Measure *body*, explaining it only when it is over budget, the one case that reads it."""
    size = len(body)
    if size <= entry.budget_bytes:
        return _Measured(entry, size)
    return _Over(entry, size, _explain_overage(entry, size, body))


def _explain_overage(entry: EndpointBudget, size: int, body: bytes) -> str:
    try:
        payload = json.loads(body)
    except ValueError:
        return (
            f'{entry.url_suffix()} served {size:,} bytes, over its '
            f'{entry.budget_bytes:,}-byte budget ({size / entry.budget_bytes:.2f}x). The body is '
            'not JSON, so no breakdown was possible.'
        )
    return render_overage(entry, size, field_breakdown(payload, entry.data_keyed_maps))


def _measure(
    client: httpx.Client, base_url: str, entry: EndpointBudget, deadline_seconds: float,
) -> _Measured | _Unmeasured:
    """Request *entry* once, abandoning it at the first body chunk past its deadline.

    Identity encoding makes the body's length the wire bytes.
    """
    give_up_at = time.monotonic() + deadline_seconds
    body = bytearray()
    try:
        with client.stream(
            'GET', f'{base_url}{entry.path}', params=dict(entry.query),
            headers={'Accept-Encoding': 'identity'},
        ) as response:
            if response.status_code != 200:
                return _Unmeasured(entry, f'HTTP {response.status_code}')
            for chunk in response.iter_bytes():
                body += chunk
                if time.monotonic() > give_up_at:
                    return _Unmeasured(entry, f'the response ran past its {deadline_seconds:g} s deadline')
    except httpx.HTTPError as exc:
        return _Unmeasured(entry, f'{type(exc).__name__}: {exc}')
    return _judged(entry, bytes(body))


def _compact(obj: object) -> str:
    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)


def _trimmed(requests: Sequence[str], keep: int) -> list[str]:
    if len(requests) <= keep:
        return list(requests)
    return [*requests[:keep], f'+{len(requests) - keep} more']


def _summary(checked: int, over: Sequence[_Over], unmeasured: Sequence[_Unmeasured], worst: _Measured | None) -> str:
    """The digest, its lists trimmed until it fits the runner's note cap."""
    over_requests = [m.entry.url_suffix() for m in sorted(over, key=lambda m: -m.ratio)]
    failed_requests = [u.entry.url_suffix() for u in unmeasured]
    max_ratio = None if worst is None else [worst.entry.url_suffix(), round(worst.ratio, 2)]
    for keep in range(max(len(over_requests), len(failed_requests)), -1, -1):
        line = _compact({
            'checked': checked,
            'over': _trimmed(over_requests, keep),
            'failed': _trimmed(failed_requests, keep),
            'max_ratio': max_ratio,
        })
        if len(line) <= SUMMARY_MAX_CHARS:
            break
    return line


def check_endpoints(
    client: httpx.Client,
    base_url: str,
    table: BudgetTable,
    *,
    deadline_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> CheckReport:
    """Measure every request in *table* against the dashboard at *base_url*, one at a time."""
    results = [
        _measure(client, base_url.rstrip('/'), entry, deadline_seconds) for entry in table.endpoints
    ]
    measured = [r for r in results if isinstance(r, _Measured)]
    unmeasured = [r for r in results if isinstance(r, _Unmeasured)]
    over = sorted((m for m in measured if isinstance(m, _Over)), key=lambda m: m.ratio)
    worst = max(measured, key=lambda m: m.ratio, default=None)
    lines = [
        *(result.line() for result in results),
        *(m.explanation for m in over),
        _summary(len(results), over, unmeasured, worst),
    ]
    exit_code = 1 if over else 2 if unmeasured else 0
    return CheckReport(exit_code=exit_code, lines=tuple(lines))


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python -m dashboard.payload_check',
        description='Measure every budgeted dashboard request against the running dashboard.',
    )
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL, help='the dashboard to measure')
    parser.add_argument(
        '--timeout', type=float, default=DEFAULT_TIMEOUT_SECONDS,
        help='seconds of wall clock allowed per request, and per network operation within it',
    )
    parser.add_argument('--table', type=Path, default=TABLE_PATH, help='the budget table to measure against')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        table = load_budget_table(args.table)
    except PayloadBudgetTableError as exc:
        print(f'payload_check: {exc}', file=sys.stderr)
        return 2
    # The dashboard is local, so no proxy from the environment applies to it.
    with httpx.Client(timeout=args.timeout, trust_env=False) as client:
        report = check_endpoints(client, args.base_url, table, deadline_seconds=args.timeout)
    print('\n'.join(report.lines))
    return report.exit_code


if __name__ == '__main__':
    sys.exit(main())
