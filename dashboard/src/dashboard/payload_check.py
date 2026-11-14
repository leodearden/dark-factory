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
  unreachable, a request timed out or answered non-200, a request was not
  reached before the run deadline, the table could not be read, or the
  checker itself raised.

An instrument failure is never reported as an overage. A measured overage
stays exit 1 even beside a request that failed, because it is a verdict
either way. When the checker itself raises, the run's verdicts are lost with
it, so that run is exit 2 whatever it had measured.

Output, in order: one line per request in table order; then each overage's
largest leaf fields, worst last; then ONE compact JSON summary line of at
most :data:`SUMMARY_MAX_CHARS`. The runner merges stdout and stderr and keeps
only their last 2,000 characters as the escalation detail, and it keeps a
trailing JSON line as the done note, so the lines that must survive come
last. Requests run one at a time: a measurement must not compete with the
next one for the dashboard.

Each request is bounded end to end, connect to last byte, by its deadline.
The run is bounded by :data:`RUN_DEADLINE_SECONDS`, so the checker always
reaches a verdict, at worst exit 2, inside :data:`PREDICATE_TIMEOUT_SECS`
however many requests the table holds. The seed link's ``timeout_secs`` is
:data:`PREDICATE_TIMEOUT_SECS`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
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
"""The per-request deadline: seconds of wall clock from connect to the last body byte."""

PREDICATE_TIMEOUT_SECS = 720
"""The before_done.timeout_secs the ``dashboard-payload-budgets`` chain is seeded with (task 5835)."""

RUN_MARGIN_SECONDS = 120.0
"""What the run leaves of the predicate's timeout: interpreter start, plus one overage breakdown in progress."""

RUN_DEADLINE_SECONDS = PREDICATE_TIMEOUT_SECS - RUN_MARGIN_SECONDS
"""Seconds of wall clock for the whole run; a request not reached by then is not measured."""

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


@dataclass(frozen=True)
class _Deadline:
    """An instant on the event loop's clock, and how a cause names the deadline that set it."""

    ends_at: float
    name: str


@dataclass(frozen=True)
class _RunClock:
    """The two wall-clock bounds of one run: each request's own, and the whole run's."""

    request_seconds: float
    run: _Deadline

    @classmethod
    def starting(cls, now: float, request_seconds: float, run_seconds: float) -> _RunClock:
        return cls(request_seconds, _Deadline(now + run_seconds, f"the run's {run_seconds:g} s deadline"))

    def next_request(self, now: float) -> _Deadline | None:
        """The deadline a request started at *now* runs under, or None once the run's has passed."""
        if now >= self.run.ends_at:
            return None
        own = _Deadline(now + self.request_seconds, f'its {self.request_seconds:g} s deadline')
        return own if own.ends_at < self.run.ends_at else self.run


async def _measure(
    client: httpx.AsyncClient, base_url: str, entry: EndpointBudget, deadline: _Deadline,
) -> _Measured | _Unmeasured:
    """Request *entry* once, connect to last byte inside *deadline*, and judge the body after it.

    Identity encoding makes the body's length the wire bytes.
    """
    body = bytearray()
    try:
        async with asyncio.timeout_at(deadline.ends_at):
            async with client.stream(
                'GET', f'{base_url}{entry.path}', params=dict(entry.query),
                headers={'Accept-Encoding': 'identity'},
            ) as response:
                if response.status_code != 200:
                    return _Unmeasured(entry, f'HTTP {response.status_code}')
                async for chunk in response.aiter_bytes():
                    body += chunk
    except TimeoutError:
        return _Unmeasured(entry, f'no complete response within {deadline.name}')
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


async def check_endpoints(
    client: httpx.AsyncClient,
    base_url: str,
    table: BudgetTable,
    *,
    request_deadline_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    run_deadline_seconds: float = RUN_DEADLINE_SECONDS,
) -> CheckReport:
    """Measure every request in *table* against the dashboard at *base_url*, one at a time, in order."""
    loop = asyncio.get_running_loop()
    clock = _RunClock.starting(loop.time(), request_deadline_seconds, run_deadline_seconds)
    results: list[_Measured | _Unmeasured] = []
    for entry in table.endpoints:
        deadline = clock.next_request(loop.time())
        if deadline is None:
            results.append(_Unmeasured(entry, f'not requested: {clock.run.name} had passed'))
        else:
            results.append(await _measure(client, base_url.rstrip('/'), entry, deadline))
    return _report(results)


def _report(results: Sequence[_Measured | _Unmeasured]) -> CheckReport:
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
        help='seconds of wall clock per request, connect to last byte',
    )
    parser.add_argument(
        '--run-deadline', type=float, default=RUN_DEADLINE_SECONDS,
        help='wall-clock seconds for the whole run; requests not reached in time are reported NOT MEASURED, exit 2',
    )
    parser.add_argument('--table', type=Path, default=TABLE_PATH, help='the budget table to measure against')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        table = load_budget_table(args.table)
        report = asyncio.run(_check_live(args.base_url, table, args.timeout, args.run_deadline))
    except PayloadBudgetTableError as exc:
        print(f'payload_check: {exc}', file=sys.stderr)
        return 2
    except Exception as exc:
        # The interpreter would exit 1 here, which is the overage code.
        traceback.print_exc()
        print(f'payload_check: instrument failure, nothing judged: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    print('\n'.join(report.lines))
    return report.exit_code


async def _check_live(
    base_url: str, table: BudgetTable, request_deadline_seconds: float, run_deadline_seconds: float,
) -> CheckReport:
    # The dashboard is local, so no proxy from the environment applies to it. The
    # client sets no timeout of its own: the two deadlines are the only bounds.
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        return await check_endpoints(
            client, base_url, table,
            request_deadline_seconds=request_deadline_seconds, run_deadline_seconds=run_deadline_seconds,
        )


if __name__ == '__main__':
    sys.exit(main())
