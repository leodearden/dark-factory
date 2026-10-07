"""The recording reverse proxy between the pinned harness and an arm (arm_harness/usage_tap.py)."""

import http.client
import json
import socket
import threading
import time
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

from _mock_openai_server import chat_completion_body, mock_openai_server

from fused_memory.arm_harness.usage_tap import (
    ERROR_EXCERPT_CHARS,
    CallRecord,
    call_record,
    load_call_records,
    usage_tap,
)

CHAT_PATH = '/v1/chat/completions'
REQUEST = {
    'model': 'qwen3.5-9b',
    'max_tokens': 4096,
    'temperature': 0.0,
    'response_format': {'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': {}}},
    'messages': [{'role': 'user', 'content': 'extract the entities'}],
}


def _post(base_url: str, path: str, body: bytes, headers: dict[str, str] | None = None):
    parts = urlsplit(base_url)
    connection = http.client.HTTPConnection(parts.hostname or '', parts.port, timeout=30)
    try:
        connection.request(
            'POST', path, body=body, headers={'Content-Type': 'application/json', **(headers or {})}
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def _origin(mock) -> str:
    return f'http://{mock.host}:{mock.port}'


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def _usage_body(prompt_tokens: int, completion_tokens: int, finish_reason: str) -> dict:
    body = chat_completion_body('{"entities": []}')
    body['usage'] = {
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': prompt_tokens + completion_tokens,
    }
    body['choices'][0]['finish_reason'] = finish_reason
    return body


def test_forwards_a_chat_completion_unchanged_and_records_the_server_usage(tmp_path):
    log = tmp_path / 'calls.jsonl'
    upstream_body = _usage_body(1234, 56, 'length')
    raw = json.dumps(REQUEST).encode()
    before = datetime.now(UTC)
    with mock_openai_server() as mock:
        mock.set_response('/chat/completions', upstream_body)
        with usage_tap(_origin(mock), log_path=log) as tap_url:
            status, body = _post(tap_url, CHAT_PATH, raw, {'Authorization': 'Bearer sk-test'})
        upstream = mock.requests_to('/chat/completions')

    assert status == 200
    assert body == json.dumps(upstream_body).encode()
    assert len(upstream) == 1
    assert upstream[0]['path'] == CHAT_PATH
    assert upstream[0]['json_body'] == REQUEST
    assert upstream[0]['headers']['Authorization'] == 'Bearer sk-test'
    assert upstream[0]['headers']['Content-Length'] == str(len(raw))

    (record,) = load_call_records(log)
    assert isinstance(record, CallRecord)
    assert before - timedelta(seconds=1) <= record.started_at <= datetime.now(UTC)
    assert record.started_at.utcoffset() == timedelta(0)
    assert record.duration_ms > 0
    assert record.method == 'POST'
    assert record.path == CHAT_PATH
    assert record.status == 200
    assert record.request_model == 'qwen3.5-9b'
    assert record.request_max_tokens == 4096
    assert record.request_response_format == 'json_schema'
    assert record.prompt_tokens == 1234
    assert record.completion_tokens == 56
    assert record.finish_reason == 'length'
    assert record.error_excerpt is None


def test_a_rejected_call_passes_through_and_records_an_excerpt_without_usage(tmp_path):
    log = tmp_path / 'calls.jsonl'
    error_body = {'error': {'message': 'maximum context length is 16384 tokens ' + 'x' * 900}}
    with mock_openai_server() as mock:
        mock.set_response_sequence('/chat/completions', [(400, error_body)])
        with usage_tap(_origin(mock), log_path=log) as tap_url:
            status, body = _post(tap_url, CHAT_PATH, json.dumps(REQUEST).encode())

    assert status == 400
    assert body == json.dumps(error_body).encode()
    (record,) = load_call_records(log)
    assert record.status == 400
    assert record.prompt_tokens is None
    assert record.completion_tokens is None
    assert record.request_model == 'qwen3.5-9b'
    assert record.error_excerpt == body.decode()[:ERROR_EXCERPT_CHARS]
    assert len(body.decode()) > ERROR_EXCERPT_CHARS


def test_a_success_without_usage_passes_through_with_no_token_counts(tmp_path):
    log = tmp_path / 'calls.jsonl'
    upstream_body = chat_completion_body('{}')
    del upstream_body['usage']
    with mock_openai_server() as mock:
        mock.set_response('/chat/completions', upstream_body)
        with usage_tap(_origin(mock), log_path=log) as tap_url:
            status, body = _post(tap_url, CHAT_PATH, json.dumps(REQUEST).encode())

    assert (status, body) == (200, json.dumps(upstream_body).encode())
    (record,) = load_call_records(log)
    assert record.prompt_tokens is None
    assert record.completion_tokens is None
    assert record.finish_reason == 'stop'


def test_call_record_tolerates_bodies_that_are_not_json():
    record = call_record(
        started_at=datetime(2026, 10, 7, tzinfo=UTC),
        duration_ms=12.5,
        method='POST',
        path=CHAT_PATH,
        request_body=b'not json either',
        status=200,
        response_body=b'<html>gateway says hi</html>',
    )

    assert record.status == 200
    assert record.request_model is None
    assert record.request_max_tokens is None
    assert record.request_response_format is None
    assert record.prompt_tokens is None
    assert record.finish_reason is None
    assert record.error_excerpt is None


def test_call_record_tolerates_json_of_the_wrong_shape():
    record = call_record(
        started_at=datetime(2026, 10, 7, tzinfo=UTC),
        duration_ms=1.0,
        method='POST',
        path=CHAT_PATH,
        request_body=json.dumps(['a', 'list']).encode(),
        status=200,
        response_body=json.dumps({'usage': 'none', 'choices': []}).encode(),
    )

    assert record.request_model is None
    assert record.prompt_tokens is None
    assert record.finish_reason is None


def test_an_unreachable_upstream_answers_502_promptly_and_names_the_error(tmp_path):
    log = tmp_path / 'calls.jsonl'
    with usage_tap(f'http://127.0.0.1:{_closed_port()}', log_path=log) as tap_url:
        started = time.monotonic()
        status, _ = _post(tap_url, CHAT_PATH, json.dumps(REQUEST).encode())
        elapsed = time.monotonic() - started

    assert status == 502
    assert elapsed < 10
    (record,) = load_call_records(log)
    assert record.status == 502
    assert record.prompt_tokens is None
    assert 'ConnectionRefusedError' in (record.error_excerpt or '')


def test_a_streamed_request_is_refused_rather_than_mismeasured(tmp_path):
    log = tmp_path / 'calls.jsonl'
    with mock_openai_server() as mock:
        with usage_tap(_origin(mock), log_path=log) as tap_url:
            status, _ = _post(tap_url, CHAT_PATH, json.dumps(REQUEST | {'stream': True}).encode())
        upstream = mock.requests

    assert status == 501
    assert upstream == []
    (record,) = load_call_records(log)
    assert record.status == 501
    assert record.prompt_tokens is None


def test_concurrent_requests_each_append_one_whole_line(tmp_path):
    log = tmp_path / 'calls.jsonl'
    statuses: list[int] = []
    with mock_openai_server() as mock:
        mock.set_response('/chat/completions', _usage_body(10, 2, 'stop'))
        with usage_tap(_origin(mock), log_path=log) as tap_url:
            def call() -> None:
                statuses.append(_post(tap_url, CHAT_PATH, json.dumps(REQUEST).encode())[0])

            threads = [threading.Thread(target=call) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)

    assert statuses == [200, 200, 200]
    lines = log.read_text().splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)['status'] == 200 for line in lines)
    assert len(load_call_records(log)) == 3


def test_records_load_in_file_order_and_the_port_closes_on_exit(tmp_path):
    log = tmp_path / 'calls.jsonl'
    with mock_openai_server() as mock:
        mock.set_response_sequence(
            '/chat/completions',
            [(200, _usage_body(100, 1, 'stop')), (200, _usage_body(200, 2, 'stop'))],
        )
        with usage_tap(_origin(mock), log_path=log) as tap_url:
            _post(tap_url, CHAT_PATH, json.dumps(REQUEST).encode())
            _post(tap_url, CHAT_PATH, json.dumps(REQUEST).encode())
    port = urlsplit(tap_url).port

    records = load_call_records(log)
    assert isinstance(records, tuple)
    assert [record.prompt_tokens for record in records] == [100, 200]
    assert records == tuple(
        CallRecord.model_validate_json(line) for line in log.read_text().splitlines()
    )
    with socket.socket() as probe:
        probe.settimeout(2)
        assert probe.connect_ex(('127.0.0.1', port)) != 0
