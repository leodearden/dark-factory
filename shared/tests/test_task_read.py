"""Tests for shared/src/shared/task_read.py::parse_get_task_result — the one classification of a
fused-memory ``get_task`` reply into TaskFound / TaskAbsent / TaskReadFailed.

The JSON-RPC body shape is the production one (payload two layers down, in
``result.content[].text`` or ``result.structuredContent``).  Only a
``TaskNotFoundError`` reply is an answer about the task; every other error reply
and every unreadable envelope means the read itself failed.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from shared.task_read import (
    TaskAbsent,
    TaskFound,
    TaskReadFailed,
    parse_get_task_result,
)


def _body(payload: object) -> dict:
    return {
        'jsonrpc': '2.0',
        'id': 1,
        'result': {'content': [{'type': 'text', 'text': json.dumps(payload)}]},
    }


def _structured_body(payload: dict) -> dict:
    return {
        'jsonrpc': '2.0',
        'id': 1,
        'result': {
            'structuredContent': payload,
            'content': [{'type': 'text', 'text': json.dumps(payload)}],
        },
    }


def _is_error_body(payload: object) -> dict:
    return {
        'jsonrpc': '2.0',
        'id': 1,
        'result': {'isError': True, 'content': [{'type': 'text', 'text': json.dumps(payload)}]},
    }


ROW = {'id': '4888', 'status': 'pending', 'metadata': {'merge_lane': 'high'}}
NOT_FOUND = {'error': 'any wording at all', 'error_type': 'TaskNotFoundError'}

ENVELOPES = [_body, _structured_body]
ENVELOPE_IDS = ['text_only', 'structured_content']


class TestTaskFound:
    @pytest.mark.parametrize('envelope', ENVELOPES, ids=ENVELOPE_IDS)
    def test_a_task_row_is_found_whole(self, envelope):
        assert parse_get_task_result(envelope(ROW)) == TaskFound(ROW)

    def test_string_metadata_is_left_exactly_as_sent(self):
        row = {'id': '7', 'status': 'done', 'metadata': '{"files": ["a.py"]}'}
        outcome = parse_get_task_result(_body(row))
        assert outcome == TaskFound(row)
        assert isinstance(outcome, TaskFound)
        assert outcome.task['metadata'] == '{"files": ["a.py"]}'

    def test_a_data_wrapper_is_unwrapped(self):
        assert parse_get_task_result(_body({'data': ROW})) == TaskFound(ROW)


class TestTaskAbsent:
    @pytest.mark.parametrize('envelope', ENVELOPES, ids=ENVELOPE_IDS)
    def test_a_task_not_found_reply_is_absence(self, envelope):
        assert parse_get_task_result(envelope(NOT_FOUND)) == TaskAbsent()


class TestTaskReadFailed:
    @pytest.mark.parametrize(
        'payload',
        [
            {'error': "INVALID_TASK_ID: non-numeric task id: 'infra'", 'error_type': 'TaskmasterError'},
            {'error': 'handler escaped', 'error_type': 'ToolError'},
            {'error': 'project_root is not a git checkout', 'error_type': 'ValidationError'},
            {'error': 'boom', 'error_type': 'BackendUnavailable'},
        ],
        ids=['invalid_task_id', 'tool_error', 'validation_error', 'backend_unavailable'],
    )
    def test_any_other_error_reply_is_a_failed_read_naming_its_type(self, payload):
        outcome = parse_get_task_result(_body(payload))
        assert isinstance(outcome, TaskReadFailed)
        assert payload['error_type'] in outcome.reason

    def test_an_error_reply_without_an_error_type_is_a_failed_read(self):
        outcome = parse_get_task_result(_body({'error': 'something broke'}))
        assert isinstance(outcome, TaskReadFailed)
        assert 'something broke' in outcome.reason

    def test_an_is_error_envelope_is_never_a_value_even_carrying_not_found(self):
        outcome = parse_get_task_result(_is_error_body(NOT_FOUND))
        assert isinstance(outcome, TaskReadFailed)

    @pytest.mark.parametrize(
        'result',
        [
            None,
            [ROW],
            {'jsonrpc': '2.0', 'id': 1, 'result': {'content': []}},
            {'jsonrpc': '2.0', 'id': 1, 'result': {'content': [{'type': 'text', 'text': 'not json'}]}},
            {'jsonrpc': '2.0', 'id': 1, 'error': {'code': -32603, 'message': 'internal error'}},
        ],
        ids=['none', 'list', 'no_text_block', 'text_not_json', 'json_rpc_error'],
    )
    def test_an_unreadable_envelope_is_a_failed_read_not_a_raise(self, result):
        assert isinstance(parse_get_task_result(result), TaskReadFailed)


class TestOutcomesAreFrozen:
    @pytest.mark.parametrize(
        ('outcome', 'field'),
        [(TaskFound(ROW), 'task'), (TaskAbsent(), 'task'), (TaskReadFailed('x'), 'reason')],
        ids=['found', 'absent', 'failed'],
    )
    def test_attribute_assignment_is_refused(self, outcome, field):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(outcome, field, None)
