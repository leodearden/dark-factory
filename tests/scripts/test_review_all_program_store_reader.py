"""``state.py::StoreReader`` against the fixture fake store.

The fake store's own truthfulness is ``test_review_all_program_fake_store.py``'s
subject; here the reader is driven through its public methods against it, and
the fake's request journal is the observable for what the reader sent (INV-11:
``include_rows`` on every find; the handshake's parity with
``scripts/migrate_metadata_modules_to_files.py::FusedMemoryClient``).
"""
from __future__ import annotations

import asyncio
import io
import json
import urllib.request
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from migrate_metadata_modules_to_files import FusedMemoryClient
from review_all_program_fixture import FIXTURE_DIR, FakeStore, load_state, running_fake_store

state = load_state()

RUN_ID = 'review-all-fixture-20261019'
PROJECT_ROOT = '/review-all-fixture/project'
PRD_PATHS = (
    'plans/review-all-fixture-s1-prd.md',
    'plans/review-all-fixture-s2-prd.md',
    'plans/review-all-fixture-s3-prd.md',
)
PROGRAM_IDS = (101, 102, 103, 104, 105, 106, 107, 108, 109, 113)

FIND_READS: dict[str, Callable[[Any], Any]] = {
    'program_tasks': lambda reader: reader.program_tasks(RUN_ID, PRD_PATHS),
    'gate': lambda reader: reader.gate(RUN_ID),
    'completion_gate': lambda reader: reader.completion_gate(RUN_ID),
    'held_leaves': lambda reader: reader.held_leaves(RUN_ID, PRD_PATHS),
    'tasks_declaring_files': lambda reader: reader.tasks_declaring_files(['src/fixture/seam.py']),
    'tasks_with_delivered_checks': lambda reader: reader.tasks_with_delivered_checks(),
}


def _store_rows(path: Path = FIXTURE_DIR / 'store.json') -> dict[int, dict[str, Any]]:
    return {task['id']: task for task in json.loads(path.read_text())['tasks']}


def _ids(read: Any) -> tuple[int, ...]:
    return tuple(row.task_id for row in read.rows)


def _reader(store: FakeStore, **kwargs: Any) -> Any:
    return state.StoreReader(store.url, PROJECT_ROOT, **kwargs)


def _tool_calls(store: FakeStore, tool: str) -> list[dict[str, Any]]:
    return [
        entry['body']['params']['arguments'] for entry in store.requests()
        if entry['body'].get('method') == 'tools/call'
        and entry['body']['params']['name'] == tool
    ]


def _raw_find(url: str, arguments: dict[str, Any]) -> dict[str, Any]:
    body = {
        'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
        'params': {'name': 'find_tasks_by_metadata', 'arguments': arguments},
    }
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream',
    })
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())['result']['structuredContent']


@pytest.fixture(scope='module')
def store() -> Iterator[FakeStore]:
    with running_fake_store() as running:
        yield running


@pytest.fixture(scope='module')
def final_store() -> Iterator[FakeStore]:
    with running_fake_store(store=FIXTURE_DIR / 'store-final.json') as running:
        yield running


def test_program_tasks_is_the_union_of_run_and_prd_path_lookups(store: FakeStore) -> None:
    read = _reader(store).program_tasks(RUN_ID, PRD_PATHS)
    assert _ids(read) == PROGRAM_IDS
    stored = _store_rows()
    assert 'x_finding_run' not in stored[104]['metadata']
    rows = {row.task_id: row for row in read.rows}
    assert rows[102].dependencies == (101,)
    assert rows[113].dependencies == (101, 106)
    assert all(row.dependencies == tuple(stored[i]['dependencies']) for i, row in rows.items())
    assert all(row.metadata == stored[i]['metadata'] for i, row in rows.items())
    measured = datetime.fromisoformat(read.measured_at)
    assert measured.utcoffset() == UTC.utcoffset(None)
    assert read.reread == ()


def test_held_leaves_are_the_unreleased_waits(store: FakeStore) -> None:
    assert _ids(_reader(store).held_leaves(RUN_ID, PRD_PATHS)) == (103,)


def test_the_interim_gate_and_no_completion_gate(store: FakeStore) -> None:
    reader = _reader(store)
    gate = reader.gate(RUN_ID)
    assert _ids(gate) == (110,)
    assert gate.rows[0].dependencies == (101, 104, 201)
    assert reader.completion_gate(RUN_ID).rows == ()


def test_the_final_form_gate_depends_on_the_completion_gate(final_store: FakeStore) -> None:
    reader = _reader(final_store)
    assert _ids(reader.completion_gate(RUN_ID)) == (111,)
    assert reader.gate(RUN_ID).rows[0].dependencies == (111,)


def test_another_runs_gates_are_never_returned(store: FakeStore) -> None:
    reader = _reader(store)
    assert reader.gate('review-all-nope-20260101').rows == ()
    assert reader.gate('review-all-fixture-20260901').rows == ()
    assert reader.gate('review-all-other-20261001').rows != ()
    assert _ids(reader.gate(RUN_ID)) == (110,)


def test_tasks_declaring_files_skip_terminal_tasks(store: FakeStore) -> None:
    assert _ids(_reader(store).tasks_declaring_files(['src/fixture/seam.py'])) == (201,)


def test_tasks_with_delivered_checks_are_the_non_terminal_ones(store: FakeStore) -> None:
    assert _ids(_reader(store).tasks_with_delivered_checks()) == (101, 202, 203, 204)


def test_task_reads_one_id_as_the_find_row(store: FakeStore) -> None:
    reader = _reader(store)
    found = {row.task_id: row for row in reader.program_tasks(RUN_ID, PRD_PATHS).rows}
    assert reader.task(101).rows == (found[101],)
    assert reader.task(999).rows == ()


def test_every_find_carries_include_rows_and_the_project_root(store: FakeStore) -> None:
    reader = _reader(store)
    for read in FIND_READS.values():
        read(reader)
    finds = _tool_calls(store, 'find_tasks_by_metadata')
    assert finds
    assert all(arguments['include_rows'] is True for arguments in finds)
    assert all(arguments['project_root'] == PROJECT_ROOT for arguments in finds)


def test_a_truncated_row_is_re_read_in_full_and_reported() -> None:
    report = io.StringIO()
    with running_fake_store('--byte-budget', '1500') as running:
        read = _reader(running, report=report).program_tasks(RUN_ID, PRD_PATHS)
        get_ids = [arguments['id'] for arguments in _tool_calls(running, 'get_task')]
        run_pages = [
            arguments for arguments in _tool_calls(running, 'find_tasks_by_metadata')
            if arguments['key'] == 'x_finding_run'
        ]
        replies = [_raw_find(running.url, arguments) for arguments in run_pages]
    assert _ids(read) == PROGRAM_IDS
    rows = {row.task_id: row for row in read.rows}
    assert rows[105].metadata == _store_rows()[105]['metadata']
    assert read.reread == (105,)
    assert get_ids == ['105']
    lines = report.getvalue().splitlines()
    assert len(lines) == 1 and '105' in lines[0]
    assert len(run_pages) > 1
    assert run_pages[0]['offset'] == 0
    for page, reply, following in zip(run_pages, replies, run_pages[1:], strict=False):
        assert following['offset'] == page['offset'] + reply['pagination']['returned']


@pytest.mark.parametrize('method', sorted(FIND_READS))
def test_a_server_without_the_widening_is_refused(method: str) -> None:
    with running_fake_store('--no-include-rows') as running:
        reader = _reader(running)
        with pytest.raises(state.StoreReaderUnsupported) as caught:
            FIND_READS[method](reader)
        assert reader.task(101).rows[0].task_id == 101
    assert caught.value.missing == 'include_rows'
    assert 'echo' in str(caught.value)


def test_a_server_that_rejects_the_argument_is_refused_the_same_way() -> None:
    with (
        running_fake_store('--no-include-rows', '--reject-unknown-args') as running,
        pytest.raises(state.StoreReaderUnsupported) as caught,
    ):
        _reader(running).program_tasks(RUN_ID, PRD_PATHS)
    assert caught.value.missing == 'include_rows'


def test_a_tool_error_that_mentions_include_rows_is_the_tools_not_inv_11s() -> None:
    text = "include_rows must be a bool; got 'yes'"
    with (
        running_fake_store('--find-error', text) as running,
        pytest.raises(state.StoreReaderError) as caught,
    ):
        _reader(running).program_tasks(RUN_ID, PRD_PATHS)
    assert not isinstance(caught.value, state.StoreReaderUnsupported)
    assert text in str(caught.value)


@pytest.mark.timeout(60)
def test_has_more_with_no_rows_is_a_reader_error_not_an_endless_loop() -> None:
    with (
        running_fake_store('--limit-cap', '0') as running,
        pytest.raises(state.StoreReaderError, match='has_more with no rows') as caught,
    ):
        _reader(running).program_tasks(RUN_ID, PRD_PATHS)
    assert not isinstance(caught.value, state.StoreReaderUnsupported)


def test_a_row_without_dependencies_is_refused_naming_key_and_task() -> None:
    with (
        running_fake_store('--drop-row-key', 'dependencies') as running,
        pytest.raises(state.StoreReaderUnsupported) as caught,
    ):
        _reader(running).program_tasks(RUN_ID, PRD_PATHS)
    assert caught.value.missing == 'dependencies'
    assert '101' in str(caught.value)


def test_sse_framing_reads_the_same_rows(store: FakeStore) -> None:
    with running_fake_store('--sse') as running:
        over_sse = _reader(running).program_tasks(RUN_ID, PRD_PATHS)
    assert over_sse.rows == _reader(store).program_tasks(RUN_ID, PRD_PATHS).rows


def test_an_unreachable_server_is_a_reader_error_naming_the_url() -> None:
    url = 'http://127.0.0.1:9/mcp'
    with pytest.raises(state.StoreReaderError, match=url):
        state.StoreReader(url, PROJECT_ROOT, timeout=5).task(101)


def test_a_url_missing_the_endpoint_is_a_reader_error_naming_the_404(store: FakeStore) -> None:
    with pytest.raises(state.StoreReaderError, match='404'):
        state.StoreReader(store.url.removesuffix('/mcp'), PROJECT_ROOT).task(101)


def _handshake(store: FakeStore) -> list[dict[str, Any]]:
    return [entry for entry in store.requests() if entry['body']['method'] != 'tools/call']


def test_the_handshake_matches_fused_memory_client(tmp_path: Path) -> None:
    async def client_handshake(base_url: str) -> None:
        async with FusedMemoryClient(base_url):
            pass

    with running_fake_store(journal=tmp_path / 'client.jsonl') as client_store:
        asyncio.run(client_handshake(client_store.url.removesuffix('/mcp')))
        client = _handshake(client_store)
    with running_fake_store(journal=tmp_path / 'reader.jsonl') as reader_store:
        _reader(reader_store).task(101)
        ours = _handshake(reader_store)

    assert [e['body']['method'] for e in client] == ['initialize', 'notifications/initialized']
    assert [e['body']['method'] for e in ours] == ['initialize', 'notifications/initialized']

    def comparable(body: dict[str, Any]) -> dict[str, Any]:
        params = dict(body['params'], clientInfo=dict(body['params']['clientInfo'], name=None))
        return dict(body, id=None, params=params)

    assert comparable(ours[0]['body']) == comparable(client[0]['body'])
    assert ours[1]['body'] == client[1]['body']
    for mine, theirs in zip(ours, client, strict=True):
        assert mine['headers']['accept'] == theirs['headers']['accept']
        assert mine['headers']['content-type'] == theirs['headers']['content-type']
        assert mine['headers']['mcp-session-id']


def test_open_reader_uses_the_program_project_root(store: FakeStore) -> None:
    program = state.parse_program(json.loads((FIXTURE_DIR / 'program.json').read_text()))
    reader = state.open_reader(program, environ={'FUSED_MEMORY_URL': store.url})
    assert _ids(reader.program_tasks(RUN_ID, PRD_PATHS)) == PROGRAM_IDS
