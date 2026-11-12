"""Tests for shared/src/shared/task_submit.py::parse_submit_task_result — the one classification of a
fused-memory ``submit_task`` response into SubmittedTicket / SubmittedTask / SubmitFailed.

The JSON-RPC body shape is the production one (payload two layers down, in
``result.content[].text``); ``shared/tests/fixtures/submit_task_responses.json`` pins the
server's own return values through the orchestrator and fused-memory suites.
"""
from __future__ import annotations

import json

import pytest

from shared.task_submit import (
    TICKET_ID_PREFIX,
    SubmitFailed,
    SubmittedTask,
    SubmittedTicket,
    is_ticket_id,
    looks_like_task_id,
    parse_submit_task_result,
)


def _body(payload: object) -> dict:
    return {
        'jsonrpc': '2.0',
        'id': 1,
        'result': {'content': [{'type': 'text', 'text': json.dumps(payload)}]},
    }


class TestSubmitted:
    def test_a_tkt_ticket_is_a_submitted_ticket(self):
        assert parse_submit_task_result(_body({'ticket': 'tkt_abc123'})) == SubmittedTicket(
            'tkt_abc123'
        )

    def test_a_task_id_is_a_submitted_task_not_combined_by_default(self):
        outcome = parse_submit_task_result(_body({'task_id': '2', 'status': 'deferred'}))
        assert outcome == SubmittedTask('2', 'deferred', combined=False)

    def test_a_combined_collision_names_the_existing_task(self):
        outcome = parse_submit_task_result(
            _body({'task_id': '7', 'status': 'done', 'combined': True})
        )
        assert outcome == SubmittedTask('7', 'done', combined=True)

    def test_a_non_string_status_arrives_as_none(self):
        outcome = parse_submit_task_result(_body({'task_id': '2', 'status': 3}))
        assert outcome == SubmittedTask('2', None)


class TestSubmitFailed:
    def test_a_ticket_without_the_tkt_prefix_is_not_a_ticket(self):
        outcome = parse_submit_task_result(_body({'ticket': 'abc123'}))
        assert isinstance(outcome, SubmitFailed)

    def test_an_int_task_id_is_not_a_task(self):
        outcome = parse_submit_task_result(_body({'task_id': 2, 'status': 'deferred'}))
        assert isinstance(outcome, SubmitFailed)

    def test_a_combine_that_names_no_task_says_so(self):
        outcome = parse_submit_task_result(
            _body({'task_id': None, 'status': None, 'combined': True})
        )
        assert isinstance(outcome, SubmitFailed)
        assert 'did not name' in outcome.reason

    def test_a_response_naming_no_task_id_is_a_failure(self):
        outcome = parse_submit_task_result(_body({'planning_mode': True}))
        assert isinstance(outcome, SubmitFailed)
        assert 'names no task id' in outcome.reason

    def test_a_structured_error_carries_its_text_and_type(self):
        outcome = parse_submit_task_result(
            _body({'error': 'backlog full', 'error_type': 'BacklogFullError'})
        )
        assert isinstance(outcome, SubmitFailed)
        assert 'backlog full' in outcome.reason
        assert 'BacklogFullError' in outcome.reason

    @pytest.mark.parametrize('result', [None, 'nope', {'task_id': '1'}], ids=['none', 'str', 'bare_dict'])
    def test_an_unwrapped_or_non_dict_result_names_the_envelope_shape(self, result):
        outcome = parse_submit_task_result(result)
        assert isinstance(outcome, SubmitFailed)
        assert 'NO_TEXT_BLOCK' in outcome.reason

    def test_the_prose_is_error_body_carries_the_prose(self):
        prose = "Error calling tool 'submit_task': 1 validation error"
        result = {
            'jsonrpc': '2.0',
            'id': 1,
            'result': {'content': [{'type': 'text', 'text': prose}], 'isError': True},
        }
        outcome = parse_submit_task_result(result)
        assert isinstance(outcome, SubmitFailed)
        assert prose in outcome.reason


class TestIdPredicates:
    def test_is_ticket_id_recognises_tkt_prefix(self):
        assert is_ticket_id('tkt_0000000000000000000000000000') is True
        assert is_ticket_id('tkt_abc') is True
        assert is_ticket_id('') is False
        assert is_ticket_id('123') is False
        assert is_ticket_id('1.2') is False
        assert is_ticket_id(None) is False

    def test_is_ticket_id_reads_the_prefix_constant(self):
        assert is_ticket_id(TICKET_ID_PREFIX + 'X') is True

    @pytest.mark.parametrize('value', ['42', '  42  ', '0', 42, 0])
    def test_looks_like_task_id_accepts_non_negative_integers(self, value):
        assert looks_like_task_id(value) is True

    @pytest.mark.parametrize('value', [True, False])
    def test_looks_like_task_id_rejects_bool(self, value):
        assert looks_like_task_id(value) is False

    @pytest.mark.parametrize('value', [-1, '-1', 'abc', '', '   ', None, 'tkt_abc', '1.5'])
    def test_looks_like_task_id_rejects_negative_and_non_numeric(self, value):
        assert looks_like_task_id(value) is False

    @pytest.mark.parametrize('value', ['²', '١٢', '４２'], ids=['superscript', 'arabic-indic', 'fullwidth'])
    def test_looks_like_task_id_rejects_non_ascii_digits(self, value):
        assert looks_like_task_id(value) is False
