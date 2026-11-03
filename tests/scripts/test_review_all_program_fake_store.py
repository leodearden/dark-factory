"""The fixture fake store tells the truth about fused-memory's MCP server.

Its oracle is raw JSON-RPC over ``urllib.request`` written here, never
``state.py``: the reader is proven against the fake in
``test_review_all_program_store_reader.py`` and the fake against the server's
documented semantics here, so a shared bug cannot hide in agreement. The
semantics mirrored are ``fused_memory/server/tools.py::find_tasks_by_metadata``
(with task 7178's ``include_rows`` widening), ``_pagination_meta`` and
``_fit_rows_to_budget``.
"""
from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from review_all_program_fixture import FIXTURE_DIR, FakeStore, running_fake_store

from shared import task_statuses

RUN_ID = 'review-all-fixture-20261019'
PROJECT_ROOT = '/review-all-fixture/project'
PROGRAM_IDS = [101, 102, 103, 105, 106, 107, 108, 109, 113]
ROW_FIELDS = {'task_id', 'title', 'status', 'priority', 'updated_at'}
NON_TERMINAL = sorted(str(s) for s in task_statuses.ACTIVE)


def _store_rows(path: Path = FIXTURE_DIR / 'store.json') -> dict[int, dict[str, Any]]:
    return {task['id']: task for task in json.loads(path.read_text())['tasks']}


def _post(
    url: str, body: Any, headers: dict[str, str] | None = None,
) -> tuple[int, str, bytes]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), method='POST',
        headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.headers.get('Content-Type', ''), response.read()
    except urllib.error.HTTPError as refused:
        return refused.code, refused.headers.get('Content-Type', ''), refused.read()


def _tools_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        'jsonrpc': '2.0', 'id': 7, 'method': 'tools/call',
        'params': {'name': name, 'arguments': arguments},
    }


def _result(url: str, name: str, **arguments: Any) -> dict[str, Any]:
    status, _, body = _post(url, _tools_call(name, arguments))
    assert status == 200, body
    return json.loads(body)['result']


def _structured(url: str, name: str, **arguments: Any) -> dict[str, Any]:
    result = _result(url, name, **arguments)
    assert not result.get('isError'), result
    return result['structuredContent']


def _find(url: str, **arguments: Any) -> dict[str, Any]:
    return _structured(url, 'find_tasks_by_metadata', project_root=PROJECT_ROOT, **arguments)


def _ids(reply: dict[str, Any]) -> list[int]:
    return [row['task_id'] for row in reply['results']]


@pytest.fixture(scope='module')
def store() -> Iterator[FakeStore]:
    with running_fake_store() as running:
        yield running


@pytest.fixture
def store_copy(tmp_path: Path) -> Path:
    copy = tmp_path / 'store.json'
    shutil.copyfile(FIXTURE_DIR / 'store.json', copy)
    return copy


def test_initialize_answers_the_server_capabilities(store: FakeStore) -> None:
    status, _, body = _post(store.url, {
        'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
        'params': {
            'protocolVersion': '2024-11-05',
            'clientInfo': {'name': 'oracle', 'version': '1.0'}, 'capabilities': {},
        },
    })
    assert status == 200
    result = json.loads(body)['result']
    assert {'protocolVersion', 'capabilities', 'serverInfo'} <= set(result)


def test_the_initialized_notification_is_accepted_without_a_body(store: FakeStore) -> None:
    status, _, body = _post(
        store.url, {'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}},
    )
    assert (status, body) == (202, b'')


def test_an_undeclared_argument_is_dropped_and_the_call_succeeds(store: FakeStore) -> None:
    reply = _find(store.url, key='x_finding_run', value=RUN_ID, bogus_arg=1)
    assert _ids(reply) == PROGRAM_IDS


def test_a_scalar_value_matches_by_equality_in_id_order(store: FakeStore) -> None:
    assert _ids(_find(store.url, key='x_finding_run', value=RUN_ID)) == PROGRAM_IDS


def test_a_list_valued_key_matches_by_membership(store: FakeStore) -> None:
    reply = _find(store.url, key='x_finding_key', value='fk-0000000000a2')
    assert _ids(reply) == [102, 104, 109]


def test_value_matching_is_type_strict(store_copy: Path) -> None:
    data = json.loads(store_copy.read_text())
    data['tasks'].append({
        'id': 300, 'title': 'carries a digit string', 'status': 'pending', 'priority': 'low',
        'updatedAt': '2026-10-19T09:00:00.000Z', 'dependencies': [],
        'metadata': {'x_ref': '101'},
    })
    store_copy.write_text(json.dumps(data))
    with running_fake_store(store=store_copy) as running:
        assert _ids(_find(running.url, key='x_ref', value=101)) == []
        assert _ids(_find(running.url, key='x_ref', value='101')) == [300]


def test_the_statuses_filter_searches_only_the_listed_statuses(store: FakeStore) -> None:
    reply = _find(store.url, key='x_finding_run', value=RUN_ID, statuses=NON_TERMINAL)
    assert _ids(reply) == [i for i in PROGRAM_IDS if i != 109]
    refused = _find(store.url, key='x_finding_run', value=RUN_ID, statuses=['finished'])
    assert refused['error_type'] == 'ValidationError'
    assert 'finished' in refused['error']


def test_the_envelope_echoes_the_query_and_include_rows(store: FakeStore) -> None:
    plain = _find(store.url, key='x_finding_key', value='fk-0000000000a2', statuses=NON_TERMINAL)
    assert (plain['key'], plain['value'], plain['statuses']) == (
        'x_finding_key', 'fk-0000000000a2', NON_TERMINAL,
    )
    assert plain['include_rows'] is False
    assert all(set(row) == ROW_FIELDS for row in plain['results'])

    full = _find(store.url, key='x_finding_run', value=RUN_ID, include_rows=True)
    assert full['include_rows'] is True
    rows = {row['task_id']: row for row in full['results']}
    stored = _store_rows()
    assert rows[113]['dependencies'] == [101, 106]
    assert all(row['metadata'] == stored[i]['metadata'] for i, row in rows.items())


def test_pagination_is_honest_and_the_limit_is_clamped(store: FakeStore) -> None:
    first = _find(store.url, key='x_finding_run', value=RUN_ID, limit=4)
    assert first['pagination'] == {
        'total': 9, 'offset': 0, 'page_size': 4, 'returned': 4, 'has_more': True,
    }
    last = _find(store.url, key='x_finding_run', value=RUN_ID, limit=4, offset=8)
    assert last['pagination'] == {
        'total': 9, 'offset': 8, 'page_size': 4, 'returned': 1, 'has_more': False,
    }
    clamped = _find(store.url, key='x_finding_run', value=RUN_ID, limit=500)
    assert clamped['pagination']['page_size'] == 100
    assert first['tasks_with_key'] == 9


def test_without_the_widening_include_rows_is_dropped_unechoed() -> None:
    with running_fake_store('--no-include-rows') as running:
        reply = _find(running.url, key='x_finding_run', value=RUN_ID, include_rows=True)
    assert 'include_rows' not in reply
    assert all(set(row) == ROW_FIELDS for row in reply['results'])


def test_a_rejecting_server_answers_is_error_naming_include_rows() -> None:
    with running_fake_store('--no-include-rows', '--reject-unknown-args') as running:
        result = _result(
            running.url, 'find_tasks_by_metadata',
            project_root=PROJECT_ROOT, key='x_finding_run', value=RUN_ID, include_rows=True,
        )
    assert result['isError'] is True
    assert 'include_rows' in result['content'][0]['text']


def test_the_byte_budget_stops_a_page_and_elides_a_lone_big_row() -> None:
    expected_bytes = len(json.dumps(_store_rows()[105]['metadata']).encode())
    with running_fake_store('--byte-budget', '1500') as running:
        before = _find(running.url, key='x_finding_run', value=RUN_ID, include_rows=True, offset=2)
        alone = _find(running.url, key='x_finding_run', value=RUN_ID, include_rows=True, offset=3)
    assert _ids(before) == [103]
    assert before['pagination']['has_more'] is True
    assert _ids(alone) == [105]
    assert alone['results'][0]['truncated_row'] is True
    assert alone['results'][0]['metadata'] == {'_elided': True, 'bytes': expected_bytes}
    assert alone['pagination']['returned'] == 1


def test_a_dropped_row_key_is_absent_from_full_rows() -> None:
    with running_fake_store('--drop-row-key', 'dependencies') as running:
        reply = _find(running.url, key='x_finding_run', value=RUN_ID, include_rows=True)
    assert reply['results']
    assert all('dependencies' not in row and 'metadata' in row for row in reply['results'])


@pytest.mark.parametrize('cap', [3, 0])
def test_the_limit_cap_clamps_every_page(cap: int) -> None:
    with running_fake_store('--limit-cap', str(cap)) as running:
        reply = _find(running.url, key='x_finding_run', value=RUN_ID, limit=50)
    assert len(reply['results']) == cap
    assert reply['pagination'] == {
        'total': 9, 'offset': 0, 'page_size': cap, 'returned': cap, 'has_more': True,
    }


def test_a_find_error_answers_every_lookup_and_leaves_get_task_alone() -> None:
    text = "include_rows must be a bool; got 'yes'"
    with running_fake_store('--find-error', text) as running:
        refused = _find(running.url, key='x_finding_run', value=RUN_ID, include_rows=True)
        task = _structured(running.url, 'get_task', id='101', project_root=PROJECT_ROOT)
    assert refused == {'error': text, 'error_type': 'ValidationError'}
    assert task['id'] == 101


def test_sse_framing_carries_the_reply_on_a_data_line() -> None:
    with running_fake_store('--sse') as running:
        status, content_type, body = _post(
            running.url,
            _tools_call('find_tasks_by_metadata', {
                'project_root': PROJECT_ROOT, 'key': 'x_finding_run', 'value': RUN_ID,
            }),
        )
    assert status == 200
    assert content_type.startswith('text/event-stream')
    data = [line[len('data:'):] for line in body.decode().splitlines() if line.startswith('data:')]
    assert len(data) == 1
    assert _ids(json.loads(data[0])['result']['structuredContent']) == PROGRAM_IDS


def test_json_framing_carries_both_text_and_structured_content(store: FakeStore) -> None:
    status, content_type, body = _post(
        store.url,
        _tools_call('find_tasks_by_metadata', {
            'project_root': PROJECT_ROOT, 'key': 'x_finding_run', 'value': RUN_ID,
        }),
    )
    assert status == 200
    assert content_type.startswith('application/json')
    result = json.loads(body)['result']
    assert result['content'][0]['type'] == 'text'
    assert json.loads(result['content'][0]['text']) == result['structuredContent']


def test_get_task_answers_the_get_task_shape(store: FakeStore) -> None:
    task = _structured(store.url, 'get_task', id='113', project_root=PROJECT_ROOT)
    stored = _store_rows()[113]
    assert task['id'] == 113
    assert (task['title'], task['status'], task['priority']) == (
        stored['title'], stored['status'], stored['priority'],
    )
    assert task['dependencies'] == [101, 106]
    assert task['metadata'] == stored['metadata']
    assert task['updatedAt'] == stored['updatedAt']


def test_get_task_of_an_unknown_id_answers_task_not_found(store: FakeStore) -> None:
    assert _structured(store.url, 'get_task', id='999', project_root=PROJECT_ROOT) == {
        'error': 'TASKMASTER_TOOL_ERROR: No tasks found for ID(s): 999',
        'error_type': 'TaskNotFoundError',
    }


def test_an_unknown_tool_or_method_answers_the_fastmcp_error_shapes(store: FakeStore) -> None:
    result = _result(store.url, 'no_such_tool')
    assert result['isError'] is True
    assert 'no_such_tool' in result['content'][0]['text']
    _, _, body = _post(store.url, {'jsonrpc': '2.0', 'id': 3, 'method': 'resources/list'})
    assert json.loads(body)['error']['code'] == -32601


def test_the_store_file_is_re_read_on_every_call(store_copy: Path) -> None:
    with running_fake_store(store=store_copy) as running:
        before = _ids(_find(running.url, key='x_finding_run', value=RUN_ID))
        data = json.loads(store_copy.read_text())
        for task in data['tasks']:
            if task['id'] == 102:
                task['metadata']['x_finding_run'] = 'review-all-other-20261001'
        store_copy.write_text(json.dumps(data))
        after = _ids(_find(running.url, key='x_finding_run', value=RUN_ID))
    assert before == PROGRAM_IDS
    assert after == [i for i in PROGRAM_IDS if i != 102]


def test_the_journal_records_each_request_as_received(tmp_path: Path) -> None:
    with running_fake_store(journal=tmp_path / 'journal.jsonl') as running:
        _post(
            running.url,
            _tools_call('find_tasks_by_metadata', {
                'project_root': PROJECT_ROOT, 'key': 'x_finding_run', 'bogus_arg': 1,
            }),
            headers={'mcp-session-id': 'abc123'},
        )
        entries = running.requests()
    assert len(entries) == 1
    entry = entries[0]
    assert entry['body']['method'] == 'tools/call'
    assert entry['body']['params']['arguments']['bogus_arg'] == 1
    assert entry['headers'] == {
        'accept': 'application/json, text/event-stream',
        'content-type': 'application/json',
        'mcp-session-id': 'abc123',
    }


def test_a_post_to_another_path_is_not_found(store: FakeStore) -> None:
    status, _, _ = _post(store.url.removesuffix('/mcp') + '/other', _tools_call('get_task', {}))
    assert status == 404
