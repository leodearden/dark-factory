"""The live payload predicate: every budgeted request, measured against the running dashboard.

``dashboard.payload_check`` is the before_done predicate of the
``dashboard-payload-budgets`` recurrence chain, so its exit code IS the
verdict: 0 within budget, 1 over, 2 the instrument failed. These tests drive
``check_endpoints`` over ``httpx.MockTransport`` and the launcher over a stub
command; nothing here reaches a network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
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
from dashboard.payload_check import check_endpoints, main, parse_args

BASE_URL = 'http://dashboard.invalid:8080'
REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / 'scripts' / 'check_dashboard_payload_budgets.sh'

SUMMARY_MAX_CHARS = 400
"""The note cap the deterministic runner applies to a predicate's trailing JSON line
(orchestrator/src/orchestrator/deterministic_runner.py::_PREDICATE_NOTE_MAX_PAYLOAD_CHARS)."""


def _table(*entries: tuple[str, Mapping[str, str], int, list[list[str]]]) -> BudgetTable:
    return parse_budget_table(json.dumps({
        'schema_version': 1,
        'endpoints': [
            {
                'path': path, 'query': dict(query), 'data_keyed_maps': keyed,
                'history': [{'budget_bytes': budget, 'task': '5824', 'reason': f'measured for {path}'}],
            }
            for path, query, budget, keyed in entries
        ],
        'exempt': [],
    }))


TABLE = _table(
    ('/api/v2/dashboard/tasks', {}, 2_000, [['TASKS_SNAPSHOT']]),
    ('/api/v2/dashboard/tasks', {'projection': 'census'}, 1_000, [['TASKS_SNAPSHOT']]),
    ('/api/v2/dashboard/burndown', {'window': '90d'}, 3_000, []),
)


def _body(size: int) -> bytes:
    """A JSON object of exactly *size* bytes, keyed by project as TASKS_SNAPSHOT is."""
    head = b'{"TASKS_SNAPSHOT":{"alpha":{"title":"'
    tail = b'"}}}'
    return head + b'x' * (size - len(head) - len(tail)) + tail


Responder = Callable[[httpx.Request], httpx.Response]


def _serving(sizes: Mapping[tuple[str, str], int]) -> Responder:
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


def _check(responder: Responder, table: BudgetTable = TABLE):
    with httpx.Client(transport=httpx.MockTransport(responder)) as client:
        return check_endpoints(client, BASE_URL, table)


def _summary(report) -> dict:
    last = report.lines[-1]
    assert len(last) <= SUMMARY_MAX_CHARS, f'the summary line is {len(last)} chars: {last}'
    assert last == json.dumps(json.loads(last), separators=(',', ':')), 'the summary line is not compact JSON'
    return json.loads(last)


def test_requests_every_entry_once_in_table_order_with_its_query():
    seen: list[tuple[str, str, str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.url.path, request.url.query.decode()))
        return _serving(WITHIN)(request)

    _check(respond)
    assert seen == [
        ('dashboard.invalid', '/api/v2/dashboard/tasks', ''),
        ('dashboard.invalid', '/api/v2/dashboard/tasks', 'projection=census'),
        ('dashboard.invalid', '/api/v2/dashboard/burndown', 'window=90d'),
    ]


def test_requests_are_sequential():
    in_flight = 0
    peak = 0

    class Counting(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                return _serving(WITHIN)(request)
            finally:
                in_flight -= 1

    with httpx.Client(transport=Counting()) as client:
        check_endpoints(client, BASE_URL, TABLE)
    assert peak == 1


def test_all_within_budget_exits_0_with_one_line_per_request_and_a_compact_summary():
    report = _check(_serving(WITHIN))

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
    assert summary['max_ratio'] == ['/api/v2/dashboard/tasks?projection=census', 0.9]


def test_an_overage_exits_1_and_reports_its_largest_fields():
    sizes = {**WITHIN, ('/api/v2/dashboard/tasks', 'projection=census'): 1_250}
    report = _check(_serving(sizes))

    assert report.exit_code == 1
    census: EndpointBudget = TABLE.endpoints[1]
    breakdown = field_breakdown(json.loads(_body(1_250)), census.data_keyed_maps)
    assert breakdown[0].path == 'TASKS_SNAPSHOT.{}.title', 'the declared data-keyed map was not collapsed'
    assert render_overage(census, 1_250, breakdown) in '\n'.join(report.lines)
    summary = _summary(report)
    assert summary['over'] == ['/api/v2/dashboard/tasks?projection=census']
    assert summary['max_ratio'] == ['/api/v2/dashboard/tasks?projection=census', 1.25]


def test_an_over_budget_body_that_is_not_json_is_still_an_overage():
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/burndown'):
            return httpx.Response(200, content=b'<html>' + b'x' * 4_000)
        return _serving(WITHIN)(request)

    report = _check(respond)

    assert report.exit_code == 1
    text = '\n'.join(report.lines)
    assert '/api/v2/dashboard/burndown?window=90d' in text
    assert '4,006' in text and '3,000' in text
    assert 'no breakdown' in text
    assert _summary(report)['over'] == ['/api/v2/dashboard/burndown?window=90d']


def _failing_on_census(failure: Callable[[httpx.Request], httpx.Response]) -> Responder:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.query == b'projection=census':
            return failure(request)
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
def test_an_instrument_failure_exits_2_and_is_never_an_overage(failure, cause):
    report = _check(_failing_on_census(failure))

    assert report.exit_code == 2
    assert any('/api/v2/dashboard/tasks?projection=census' in line and cause in line for line in report.lines)
    text = '\n'.join(report.lines)
    assert 'against a budget of' not in text, 'an instrument failure was reported as an overage'
    summary = _summary(report)
    assert summary['over'] == []
    assert summary['failed'] == ['/api/v2/dashboard/tasks?projection=census']


def test_an_overage_beside_an_instrument_failure_is_still_an_overage():
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/burndown'):
            return httpx.Response(200, content=_body(3_300))
        return _failing_on_census(_refused)(request)

    report = _check(respond)

    assert report.exit_code == 1
    summary = _summary(report)
    assert summary['over'] == ['/api/v2/dashboard/burndown?window=90d']
    assert summary['failed'] == ['/api/v2/dashboard/tasks?projection=census']


def test_the_summary_fits_the_note_cap_however_many_requests_are_over():
    many = _table(*(
        (f'/api/v2/dashboard/an-endpoint-with-a-long-name-{n:02d}', {'window': 'all'}, 10, [])
        for n in range(30)
    ))

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_body(500))

    report = _check(respond, many)

    assert report.exit_code == 1
    summary = _summary(report)
    assert summary['checked'] == 30
    assert summary['over'], 'the summary dropped every overage instead of trimming the list'
    assert summary['over'][-1].startswith('+'), 'a trimmed list must say how many it left out'


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
    assert args.table == TABLE_PATH

    args = parse_args(['--base-url', BASE_URL, '--timeout', '5', '--table', '/tmp/t.json'])
    assert (args.base_url, args.timeout, args.table) == (BASE_URL, 5.0, Path('/tmp/t.json'))


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
