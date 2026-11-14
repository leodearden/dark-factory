"""The live payload predicate: every budgeted request, measured against the running dashboard.

``dashboard.payload_check`` is the before_done predicate of the
``dashboard-payload-budgets`` recurrence chain, so its exit code IS the
verdict: 0 within budget, 1 over, 2 the instrument failed. These tests drive
``check_endpoints`` over ``httpx.MockTransport``, ``main`` over a loopback
socket that never accepts, and the launcher over a stub command; nothing here
reaches beyond the host.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from pathlib import Path

import httpx
import pytest

from dashboard.payload_budgets import (
    TABLE_PATH,
    BudgetTable,
    EndpointBudget,
    field_breakdown,
    parse_budget_table,
    render_overage,
)
from dashboard.payload_check import (
    DEFAULT_TIMEOUT_SECONDS,
    RUN_DEADLINE_SECONDS,
    SUMMARY_MAX_CHARS,
    check_endpoints,
    main,
    parse_args,
)

BASE_URL = 'http://dashboard.invalid:8080'
REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / 'scripts' / 'check_dashboard_payload_budgets.sh'

STALL_SECONDS = 30.0
"""Far longer than any deadline under test, so a stalled request ends only by a deadline."""

BOUNDED_SLACK_SECONDS = 4.0
"""How far past its bound a deadline-bounded call may return on a loaded host."""

TableEntry = tuple[str, Mapping[str, str], int, list[list[str]]]


def _table_document(*entries: TableEntry) -> dict:
    return {
        'schema_version': 1,
        'endpoints': [
            {
                'path': path, 'query': dict(query), 'data_keyed_maps': keyed,
                'history': [{'budget_bytes': budget, 'task': '5824', 'reason': f'measured for {path}'}],
            }
            for path, query, budget, keyed in entries
        ],
        'exempt': [],
    }


def _table(*entries: TableEntry) -> BudgetTable:
    return parse_budget_table(json.dumps(_table_document(*entries)))


TABLE_ENTRIES: tuple[TableEntry, ...] = (
    ('/api/v2/dashboard/tasks', {}, 2_000, [['TASKS_SNAPSHOT']]),
    ('/api/v2/dashboard/tasks', {'projection': 'census'}, 1_000, [['TASKS_SNAPSHOT']]),
    ('/api/v2/dashboard/burndown', {'window': '90d'}, 3_000, []),
)
TABLE = _table(*TABLE_ENTRIES)
TASKS, CENSUS, BURNDOWN = (entry.url_suffix() for entry in TABLE.endpoints)


def _body(size: int) -> bytes:
    """A JSON object of exactly *size* bytes, keyed by project as TASKS_SNAPSHOT is."""
    head = b'{"TASKS_SNAPSHOT":{"alpha":{"title":"'
    tail = b'"}}}'
    return head + b'x' * (size - len(head) - len(tail)) + tail


SyncResponder = Callable[[httpx.Request], httpx.Response]
AsyncResponder = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
Responder = SyncResponder | AsyncResponder


def _serving(sizes: Mapping[tuple[str, str], int]) -> SyncResponder:
    """Answer each (path, query string) with a JSON body of the given size."""
    def respond(request: httpx.Request) -> httpx.Response:
        size = sizes[(request.url.path, request.url.query.decode())]
        return httpx.Response(200, content=_body(size), headers={'content-type': 'application/json'})
    return respond


WITHIN = {
    ('/api/v2/dashboard/tasks', ''): 1_500,
    ('/api/v2/dashboard/tasks', 'projection=census'): 900,
    ('/api/v2/dashboard/burndown', 'window=90d'): 1_200,
}


async def _stall(_request: httpx.Request) -> httpx.Response:
    """Send no headers for far longer than any deadline under test."""
    await asyncio.sleep(STALL_SECONDS)
    raise AssertionError('a stalled request outlived every deadline')


async def _check(
    responder: Responder,
    table: BudgetTable = TABLE,
    *,
    request_deadline_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    run_deadline_seconds: float = RUN_DEADLINE_SECONDS,
):
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        return await check_endpoints(
            client, BASE_URL, table,
            request_deadline_seconds=request_deadline_seconds,
            run_deadline_seconds=run_deadline_seconds,
        )


async def _timed_check(responder: Responder, table: BudgetTable = TABLE, **deadlines: float):
    started = time.monotonic()
    report = await _check(responder, table, **deadlines)
    return report, time.monotonic() - started


def _summary(report) -> dict:
    last = report.lines[-1]
    assert len(last) <= SUMMARY_MAX_CHARS, f'the summary line is {len(last)} chars: {last}'
    assert last == json.dumps(json.loads(last), separators=(',', ':')), 'the summary line is not compact JSON'
    return json.loads(last)


def _accounted(listed: Sequence[str]) -> int:
    """How many requests a summary list stands for, counting a trailing '+N more'."""
    if listed and listed[-1].startswith('+'):
        return len(listed) - 1 + int(listed[-1][1:].split()[0])
    return len(listed)


def _line_for(report, suffix: str) -> str:
    return next(line for line in report.lines if line.startswith(f'{suffix}: '))


async def test_requests_every_entry_once_in_table_order_with_its_query():
    seen: list[tuple[str, str, str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.url.path, request.url.query.decode()))
        return _serving(WITHIN)(request)

    await _check(respond)
    assert seen == [
        ('dashboard.invalid', '/api/v2/dashboard/tasks', ''),
        ('dashboard.invalid', '/api/v2/dashboard/tasks', 'projection=census'),
        ('dashboard.invalid', '/api/v2/dashboard/burndown', 'window=90d'),
    ]


async def test_requests_are_sequential():
    in_flight = 0
    peak = 0

    class Counting(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0)
                return _serving(WITHIN)(request)
            finally:
                in_flight -= 1

    async with httpx.AsyncClient(transport=Counting()) as client:
        await check_endpoints(client, BASE_URL, TABLE)
    assert peak == 1


async def test_all_within_budget_exits_0_with_one_line_per_request_and_a_compact_summary():
    report = await _check(_serving(WITHIN))

    assert report.exit_code == 0
    for entry, measured in zip(TABLE.endpoints, WITHIN.values(), strict=True):
        assert any(
            entry.url_suffix() in line and f'{measured:,}' in line and f'{entry.budget_bytes:,}' in line
            for line in report.lines
        ), f'no measured/budget line for {entry.url_suffix()}'
    summary = _summary(report)
    assert summary['checked'] == 3
    assert summary['over'] == []
    assert summary['failed'] == []
    assert summary['max_ratio'] == [CENSUS, 0.9]


async def test_an_overage_exits_1_and_reports_its_largest_fields():
    sizes = {**WITHIN, ('/api/v2/dashboard/tasks', 'projection=census'): 1_250}
    report = await _check(_serving(sizes))

    assert report.exit_code == 1
    census: EndpointBudget = TABLE.endpoints[1]
    breakdown = field_breakdown(json.loads(_body(1_250)), census.data_keyed_maps)
    assert breakdown[0].path == 'TASKS_SNAPSHOT.{}.title', 'the declared data-keyed map was not collapsed'
    assert render_overage(census, 1_250, breakdown) in '\n'.join(report.lines)
    summary = _summary(report)
    assert summary['over'] == [CENSUS]
    assert summary['max_ratio'] == [CENSUS, 1.25]


async def test_an_over_budget_body_that_is_not_json_is_still_an_overage():
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/burndown'):
            return httpx.Response(200, content=b'<html>' + b'x' * 4_000)
        return _serving(WITHIN)(request)

    report = await _check(respond)

    assert report.exit_code == 1
    text = '\n'.join(report.lines)
    assert BURNDOWN in text
    assert '4,006' in text and '3,000' in text
    assert 'no breakdown' in text
    assert _summary(report)['over'] == [BURNDOWN]


def _failing_on_census(failure: Responder) -> AsyncResponder:
    async def respond(request: httpx.Request) -> httpx.Response:
        if request.url.query == b'projection=census':
            answer = failure(request)
            return answer if isinstance(answer, httpx.Response) else await answer
        return _serving(WITHIN)(request)
    return respond


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError('connection refused', request=request)


def _timed_out(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout('read timed out', request=request)


def _server_error(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, content=_body(5_000))


@pytest.mark.parametrize(('failure', 'cause'), [
    (_refused, 'connection refused'),
    (_timed_out, 'read timed out'),
    (_server_error, '503'),
], ids=['connect-error', 'timeout', 'non-200'])
async def test_an_instrument_failure_exits_2_and_is_never_an_overage(failure, cause):
    report = await _check(_failing_on_census(failure))

    assert report.exit_code == 2
    assert any(CENSUS in line and cause in line for line in report.lines)
    text = '\n'.join(report.lines)
    assert 'against a budget of' not in text, 'an instrument failure was reported as an overage'
    summary = _summary(report)
    assert summary['over'] == []
    assert summary['failed'] == [CENSUS]


async def test_a_response_trickling_past_its_deadline_is_abandoned_unmeasured():
    chunks = 30
    sent = 0

    async def trickle():
        nonlocal sent
        for _ in range(chunks):
            await asyncio.sleep(0.1)
            sent += 1
            yield b' '

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/burndown'):
            return httpx.Response(200, content=trickle())
        return _serving(WITHIN)(request)

    report = await _check(respond, request_deadline_seconds=0.5)

    assert sent < chunks, 'the trickling body was read to its end instead of abandoned at the deadline'
    assert report.exit_code == 2
    assert any(BURNDOWN in line and 'deadline' in line for line in report.lines)
    assert _summary(report)['failed'] == [BURNDOWN]


async def test_headers_that_never_arrive_are_bounded_by_the_request_deadline():
    report, elapsed = await _timed_check(_failing_on_census(_stall), request_deadline_seconds=0.3)

    assert elapsed < 0.3 + BOUNDED_SLACK_SECONDS, f'a request with no headers held the run for {elapsed:.1f} s'
    assert report.exit_code == 2
    stalled = _line_for(report, CENSUS)
    assert 'NOT MEASURED' in stalled and '0.3 s deadline' in stalled and "run's" not in stalled, stalled
    for suffix in (TASKS, BURNDOWN):
        assert 'NOT MEASURED' not in _line_for(report, suffix), 'a stall in one request cost another its measurement'
    assert _summary(report)['failed'] == [CENSUS]


async def test_the_run_deadline_bounds_the_run_however_many_requests_stall():
    stalled_table = _table(*((f'/api/v2/dashboard/stalled-{n}', {}, 1_000, []) for n in range(8)))
    started: list[str] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        started.append(request.url.path)
        return await _stall(request)

    report, elapsed = await _timed_check(
        respond, stalled_table, request_deadline_seconds=0.4, run_deadline_seconds=1.0,
    )

    assert elapsed < 1.0 + BOUNDED_SLACK_SECONDS, f'the run took {elapsed:.1f} s against a 1 s run deadline'
    assert report.exit_code == 2
    assert 1 <= len(started) <= 3, f'{len(started)} requests started inside a 1 s run of 0.4 s requests'
    for entry in stalled_table.endpoints:
        line = _line_for(report, entry.url_suffix())
        assert 'NOT MEASURED' in line, line
        if entry.path in started:
            assert 'not requested' not in line, line
            assert '0.4 s deadline' in line or "the run's 1 s deadline" in line, line
        else:
            assert "not requested: the run's 1 s deadline had passed" in line, line
    assert _accounted(_summary(report)['failed']) == 8


async def test_a_request_cut_short_by_the_run_deadline_names_the_runs_deadline():
    report, elapsed = await _timed_check(
        _stall, _table(TABLE_ENTRIES[0]), request_deadline_seconds=STALL_SECONDS, run_deadline_seconds=0.3,
    )

    assert elapsed < 0.3 + BOUNDED_SLACK_SECONDS
    assert report.exit_code == 2
    line = _line_for(report, TASKS)
    assert "the run's 0.3 s deadline" in line and 'not requested' not in line, line
    assert f'{STALL_SECONDS:g} s deadline' not in line, line


async def test_an_overage_outlives_the_run_deadline():
    async def respond(request: httpx.Request) -> httpx.Response:
        if (request.url.path, request.url.query) == ('/api/v2/dashboard/tasks', b''):
            return httpx.Response(200, content=_body(2_500))
        return await _stall(request)

    report, elapsed = await _timed_check(respond, run_deadline_seconds=0.5)

    assert elapsed < 0.5 + BOUNDED_SLACK_SECONDS
    assert report.exit_code == 1, 'the run deadline cost a measured overage its verdict'
    tasks: EndpointBudget = TABLE.endpoints[0]
    breakdown = field_breakdown(json.loads(_body(2_500)), tasks.data_keyed_maps)
    assert render_overage(tasks, 2_500, breakdown) in '\n'.join(report.lines)
    summary = _summary(report)
    assert summary['over'] == [TASKS]
    assert summary['failed'] == [CENSUS, BURNDOWN]


def test_main_bounds_each_request_and_the_run_over_a_real_stalled_socket(tmp_path, capsys):
    table = tmp_path / 'payload_budgets.json'
    table.write_text(json.dumps(_table_document(*TABLE_ENTRIES)))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(8)
        port = listener.getsockname()[1]
        started = time.monotonic()
        code = main([
            '--base-url', f'http://127.0.0.1:{port}', '--timeout', '0.3', '--run-deadline', '0.8',
            '--table', str(table),
        ])
        elapsed = time.monotonic() - started

    out = capsys.readouterr().out
    assert code == 2, out
    assert elapsed < 0.8 + BOUNDED_SLACK_SECONDS, f'main took {elapsed:.1f} s against a 0.8 s run deadline'
    assert any("the run's 0.8 s deadline" in line for line in out.splitlines()), out


def test_an_exception_the_checker_does_not_anticipate_exits_2_naming_it(tmp_path, capsys):
    """httpx.InvalidURL is not an httpx.HTTPError, so no request-level cause catches it."""
    table = tmp_path / 'payload_budgets.json'
    table.write_text(json.dumps(_table_document(*TABLE_ENTRIES)))

    code = main(['--base-url', 'http://[::1', '--timeout', '1', '--run-deadline', '2', '--table', str(table)])

    captured = capsys.readouterr()
    assert code == 2, captured.out + captured.err
    last = captured.err.rstrip().splitlines()[-1]
    assert 'instrument failure' in last and 'InvalidURL' in last, captured.err
    assert 'against a budget of' not in captured.out + captured.err


@pytest.mark.parametrize('content', [None, '{"schema_version": 1, "endpoints": []}'], ids=['missing', 'malformed'])
def test_an_unreadable_table_exits_2(tmp_path, capsys, content):
    table = tmp_path / 'payload_budgets.json'
    if content is not None:
        table.write_text(content)

    assert main(['--table', str(table), '--base-url', BASE_URL]) == 2
    captured = capsys.readouterr()
    assert str(table) in captured.out + captured.err


def test_cli_defaults():
    args = parse_args([])
    assert args.base_url == 'http://127.0.0.1:8080'
    assert args.timeout == 60.0
    assert args.run_deadline == RUN_DEADLINE_SECONDS
    assert args.table == TABLE_PATH

    args = parse_args([
        '--base-url', BASE_URL, '--timeout', '5', '--run-deadline', '30', '--table', '/tmp/t.json',
    ])
    assert (args.base_url, args.timeout, args.run_deadline, args.table) == (
        BASE_URL, 5.0, 30.0, Path('/tmp/t.json'),
    )


def test_the_launcher_is_tracked_executable():
    listed = subprocess.run(
        ['git', 'ls-files', '-s', '--', str(LAUNCHER.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    assert listed.startswith('100755 '), (
        f'{LAUNCHER} must be tracked with mode 100755 (git update-index --chmod=+x); got {listed!r}'
    )


def _run_launcher(tmp_path: Path, *args: str, stub_exit: int) -> subprocess.CompletedProcess[str]:
    stub = tmp_path / 'stub.py'
    stub.write_text(
        'import os, sys\n'
        "sys.stdout.write('\\n'.join(sys.argv[1:]))\n"
        "sys.exit(int(os.environ['STUB_EXIT']))\n"
    )
    env = {**os.environ, 'CHECK_DASHBOARD_PAYLOAD_PY': f'{sys.executable} {stub}', 'STUB_EXIT': str(stub_exit)}
    return subprocess.run([str(LAUNCHER), *args], capture_output=True, text=True, env=env, timeout=30)


def test_the_launcher_forwards_its_arguments_unchanged(tmp_path):
    result = _run_launcher(tmp_path, '--base-url', BASE_URL, '--timeout', '7', stub_exit=0)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['--base-url', BASE_URL, '--timeout', '7']


@pytest.mark.parametrize('code', [0, 1, 2])
def test_the_launcher_propagates_the_exit_code(tmp_path, code):
    assert _run_launcher(tmp_path, stub_exit=code).returncode == code
