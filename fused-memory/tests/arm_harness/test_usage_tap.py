"""The recording reverse proxy between the pinned harness and an arm (arm_harness/usage_tap.py)."""

import http.client
import io
import json
import signal
import socket
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, closing, redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from _mock_openai_server import chat_completion_body, mock_openai_server

from arm_harness._loopback_http import (
    HELD_PROMPT_TOKENS,
    held_upstream,
    post_json,
    unused_port,
    usage_body,
)
from fused_memory.arm_harness.usage_tap import (
    ERROR_EXCERPT_CHARS,
    CallRecord,
    StatusOrigin,
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
DRAIN_WAITING = 'usage tap: waiting'


def _origin(mock) -> str:
    return f'http://{mock.host}:{mock.port}'


def test_forwards_a_chat_completion_unchanged_and_records_the_server_usage(tmp_path):
    log = tmp_path / 'calls.jsonl'
    upstream_body = usage_body(1234, 56, 'length')
    raw = json.dumps(REQUEST).encode()
    before = datetime.now(UTC)
    with mock_openai_server() as mock:
        mock.set_response('/chat/completions', upstream_body)
        with usage_tap(_origin(mock), log_path=log) as tap_url:
            status, body = post_json(tap_url, CHAT_PATH, raw, {'Authorization': 'Bearer sk-test'})
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
    assert record.outlived_session is False
    assert record.status_origin is StatusOrigin.UPSTREAM


def test_a_rejected_call_passes_through_and_records_an_excerpt_without_usage(tmp_path):
    log = tmp_path / 'calls.jsonl'
    error_body = {'error': {'message': 'maximum context length is 16384 tokens ' + 'x' * 900}}
    with mock_openai_server() as mock:
        mock.set_response_sequence('/chat/completions', [(400, error_body)])
        with usage_tap(_origin(mock), log_path=log) as tap_url:
            status, body = post_json(tap_url, CHAT_PATH, json.dumps(REQUEST).encode())

    assert status == 400
    assert body == json.dumps(error_body).encode()
    (record,) = load_call_records(log)
    assert record.status == 400
    assert record.prompt_tokens is None
    assert record.completion_tokens is None
    assert record.request_model == 'qwen3.5-9b'
    assert record.error_excerpt == body.decode()[:ERROR_EXCERPT_CHARS]
    assert len(body.decode()) > ERROR_EXCERPT_CHARS
    assert record.status_origin is StatusOrigin.UPSTREAM


def test_a_success_without_usage_passes_through_with_no_token_counts(tmp_path):
    log = tmp_path / 'calls.jsonl'
    upstream_body = chat_completion_body('{}')
    del upstream_body['usage']
    with mock_openai_server() as mock:
        mock.set_response('/chat/completions', upstream_body)
        with usage_tap(_origin(mock), log_path=log) as tap_url:
            status, body = post_json(tap_url, CHAT_PATH, json.dumps(REQUEST).encode())

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
        status_origin=StatusOrigin.UPSTREAM,
    )

    assert record.status == 200
    assert record.request_model is None
    assert record.request_max_tokens is None
    assert record.request_response_format is None
    assert record.prompt_tokens is None
    assert record.finish_reason is None
    assert record.error_excerpt is None
    assert record.status_origin is StatusOrigin.UPSTREAM


def test_call_record_tolerates_json_of_the_wrong_shape():
    record = call_record(
        started_at=datetime(2026, 10, 7, tzinfo=UTC),
        duration_ms=1.0,
        method='POST',
        path=CHAT_PATH,
        request_body=json.dumps(['a', 'list']).encode(),
        status=200,
        response_body=json.dumps({'usage': 'none', 'choices': []}).encode(),
        status_origin=StatusOrigin.UPSTREAM,
    )

    assert record.request_model is None
    assert record.prompt_tokens is None
    assert record.finish_reason is None
    assert record.status_origin is StatusOrigin.UPSTREAM


def test_an_unreachable_upstream_answers_502_promptly_and_names_the_error(tmp_path):
    log = tmp_path / 'calls.jsonl'
    with usage_tap(f'http://127.0.0.1:{unused_port()}', log_path=log) as tap_url:
        started = time.monotonic()
        status, _ = post_json(tap_url, CHAT_PATH, json.dumps(REQUEST).encode())
        elapsed = time.monotonic() - started

    assert status == 502
    assert elapsed < 10
    (record,) = load_call_records(log)
    assert record.status == 502
    assert record.prompt_tokens is None
    assert 'ConnectionRefusedError' in (record.error_excerpt or '')
    assert record.status_origin is StatusOrigin.TAP


def test_a_streamed_request_is_refused_rather_than_mismeasured(tmp_path):
    log = tmp_path / 'calls.jsonl'
    with mock_openai_server() as mock:
        with usage_tap(_origin(mock), log_path=log) as tap_url:
            streamed = json.dumps(REQUEST | {'stream': True}).encode()
            status, _ = post_json(tap_url, CHAT_PATH, streamed)
        upstream = mock.requests

    assert status == 501
    assert upstream == []
    (record,) = load_call_records(log)
    assert record.status == 501
    assert record.prompt_tokens is None
    assert record.status_origin is StatusOrigin.TAP


def test_concurrent_requests_each_append_one_whole_line(tmp_path):
    log = tmp_path / 'calls.jsonl'
    statuses: list[int] = []
    with mock_openai_server() as mock:
        mock.set_response('/chat/completions', usage_body(10, 2, 'stop'))
        with usage_tap(_origin(mock), log_path=log) as tap_url:
            def call() -> None:
                statuses.append(post_json(tap_url, CHAT_PATH, json.dumps(REQUEST).encode())[0])

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
            [(200, usage_body(100, 1, 'stop')), (200, usage_body(200, 2, 'stop'))],
        )
        with usage_tap(_origin(mock), log_path=log) as tap_url:
            post_json(tap_url, CHAT_PATH, json.dumps(REQUEST).encode())
            post_json(tap_url, CHAT_PATH, json.dumps(REQUEST).encode())
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


def test_a_session_with_no_calls_still_leaves_its_empty_log(tmp_path, capsys):
    log = tmp_path / 'calls.jsonl'
    with mock_openai_server() as mock, usage_tap(_origin(mock), log_path=log):
        pass

    assert log.read_text() == ''
    assert load_call_records(log) == ()
    assert capsys.readouterr().out == ''


def _post_in_background(tap_url: str, statuses: list[int]) -> threading.Thread:
    raw = json.dumps(REQUEST).encode()
    client = threading.Thread(
        target=lambda: statuses.append(post_json(tap_url, CHAT_PATH, raw)[0]), daemon=True
    )
    client.start()
    return client


class _DrainNoticeWatcher(io.StringIO):
    """Stdout that sets ``draining`` once the closing session announces it is draining."""

    def __init__(self, draining: threading.Event) -> None:
        super().__init__()
        self._draining = draining

    def write(self, text: str) -> int:
        written = super().write(text)
        if DRAIN_WAITING in self.getvalue():
            self._draining.set()
        return written


def test_closing_a_session_waits_for_a_call_still_in_flight_and_records_it_inside_the_session(
    tmp_path,
):
    log = tmp_path / 'calls.jsonl'
    received, release = threading.Event(), threading.Event()
    statuses: list[int] = []
    with held_upstream(received, release) as upstream:
        stdout = _DrainNoticeWatcher(draining=release)
        with redirect_stdout(stdout), usage_tap(upstream, log_path=log) as tap_url:
            client = _post_in_background(tap_url, statuses)
            assert received.wait(timeout=10)
        recorded_at_session_exit = load_call_records(log)
        client.join(timeout=10)

    (record,) = recorded_at_session_exit
    assert record.status == 200
    assert record.prompt_tokens == HELD_PROMPT_TOKENS
    assert record.outlived_session is False
    assert statuses == [200]
    (waiting,) = stdout.getvalue().splitlines()
    assert waiting.startswith(DRAIN_WAITING)
    assert '1 in-flight call' in waiting


def test_a_call_still_in_flight_when_the_drain_bound_expires_is_marked_as_outliving_the_session(
    tmp_path, capsys
):
    log = tmp_path / 'calls.jsonl'
    received, release = threading.Event(), threading.Event()
    statuses: list[int] = []
    with held_upstream(received, release) as upstream:
        with usage_tap(upstream, log_path=log, drain_timeout_s=0.2) as tap_url:
            client = _post_in_background(tap_url, statuses)
            assert received.wait(timeout=10)
            exiting = time.monotonic()
        exit_s = time.monotonic() - exiting
        recorded_at_session_exit = log.read_text()
        release.set()
        client.join(timeout=10)

    assert exit_s < 5
    assert recorded_at_session_exit == ''
    assert statuses == [200]
    (record,) = load_call_records(log)
    assert record.outlived_session is True
    assert record.status == 200
    assert record.prompt_tokens == HELD_PROMPT_TOKENS
    waiting, expired = capsys.readouterr().out.splitlines()
    assert waiting.startswith(DRAIN_WAITING)
    assert expired.startswith('usage tap: drain bound')
    assert '1 call(s) still in flight' in expired


class _InterruptedWhileDraining(Exception):
    """Stands in for a KeyboardInterrupt that lands while a closing session drains."""


def _raise_interrupted(signum, frame) -> None:
    raise _InterruptedWhileDraining


def _signal_the_main_thread() -> None:
    signal.pthread_kill(threading.main_thread().ident or 0, signal.SIGUSR1)


def _signal_this_thread() -> None:
    """The handler then waits on the main thread, whose blocked wait gets no EINTR."""
    signal.pthread_kill(threading.get_ident(), signal.SIGUSR1)


def _interrupt_once_draining(
    draining: threading.Event, deliver: Callable[[], None]
) -> threading.Thread:
    """Deliver an interrupt once the closing session announces it is draining."""

    def interrupt() -> None:
        if draining.wait(timeout=10):
            deliver()

    interrupter = threading.Thread(target=interrupt, daemon=True)
    interrupter.start()
    return interrupter


def _drain_interrupted_by(log: Path, deliver: Callable[[], None]) -> list[int]:
    """Interrupt a session draining one held call, then release it; the client's statuses."""
    received, release, draining = threading.Event(), threading.Event(), threading.Event()
    statuses: list[int] = []
    previous = signal.signal(signal.SIGUSR1, _raise_interrupted)
    try:
        with held_upstream(received, release) as upstream:
            with pytest.raises(_InterruptedWhileDraining), redirect_stdout(
                _DrainNoticeWatcher(draining)
            ), usage_tap(upstream, log_path=log) as tap_url:
                client = _post_in_background(tap_url, statuses)
                assert received.wait(timeout=10)
                interrupter = _interrupt_once_draining(draining, deliver)
            interrupter.join(timeout=10)
            release.set()
            client.join(timeout=10)
    finally:
        signal.signal(signal.SIGUSR1, previous)
    return statuses


def test_an_interrupted_drain_still_marks_the_abandoned_call_as_outliving_the_session(tmp_path):
    log = tmp_path / 'calls.jsonl'
    statuses = _drain_interrupted_by(log, _signal_the_main_thread)

    assert statuses == [200]
    (record,) = load_call_records(log)
    assert record.outlived_session is True


def test_an_interrupt_delivered_to_another_thread_still_ends_the_drain(tmp_path):
    log = tmp_path / 'calls.jsonl'
    statuses = _drain_interrupted_by(log, _signal_this_thread)

    assert statuses == [200]
    (record,) = load_call_records(log)
    assert record.outlived_session is True


def test_an_idle_keep_alive_connection_does_not_hold_a_closing_session_open(tmp_path):
    log = tmp_path / 'calls.jsonl'
    with mock_openai_server() as mock, ExitStack() as connections:
        mock.set_response('/chat/completions', usage_body(10, 2, 'stop'))
        with usage_tap(_origin(mock), log_path=log) as tap_url:
            parts = urlsplit(tap_url)
            connection = connections.enter_context(
                closing(http.client.HTTPConnection(parts.hostname or '', parts.port, timeout=30))
            )
            connection.request(
                'POST', CHAT_PATH, body=json.dumps(REQUEST).encode(),
                headers={'Content-Type': 'application/json'},
            )
            response = connection.getresponse()
            response.read()
            exiting = time.monotonic()
        exit_s = time.monotonic() - exiting

    assert response.status == 200
    assert exit_s < 5


def test_a_record_predating_the_session_and_origin_fields_loads_in_session_with_no_origin(tmp_path):
    log = tmp_path / 'calls.jsonl'
    record = call_record(
        started_at=datetime(2026, 10, 7, tzinfo=UTC),
        duration_ms=1.0,
        method='POST',
        path=CHAT_PATH,
        request_body=json.dumps(REQUEST).encode(),
        status=200,
        response_body=json.dumps(usage_body(10, 1, 'stop')).encode(),
        status_origin=StatusOrigin.UPSTREAM,
    )
    payload = record.model_dump(mode='json')
    del payload['outlived_session']
    del payload['status_origin']
    log.write_text(json.dumps(payload) + '\n')

    (loaded,) = load_call_records(log)
    assert loaded.outlived_session is False
    assert loaded.status_origin is None
    assert loaded == record.model_copy(update={'status_origin': None})
