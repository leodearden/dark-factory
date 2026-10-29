"""``parse_claude_cli_output`` reads one finished ``claude --print`` result.

It is the reader for callers that ran the CLI themselves, such as
``shared.usage_gate.UsageGate._run_probe``. These tests drive only that public
seam and assert on the returned ``AgentResult``'s public fields.
"""

from __future__ import annotations

import json

import pytest
from _measured_cli_results import MEASURED_ORG_DISABLED_403, ORG_DISABLED_TEXT

from shared.cli_invoke import BUDGET_EXHAUSTED_SUBTYPE, parse_claude_cli_output


def test_served_result_is_success() -> None:
    stdout = json.dumps({
        'type': 'result',
        'subtype': 'success',
        'is_error': False,
        'result': 'ok',
        'num_turns': 1,
        'total_cost_usd': 0.002,
    })
    result = parse_claude_cli_output(stdout=stdout, stderr='', returncode=0)
    assert result.success is True
    assert result.subtype == 'success'
    assert result.output == 'ok'
    assert result.api_error_status is None
    assert result.schema_salvaged is False


@pytest.mark.parametrize('returncode', [0, 1])
def test_is_error_overrides_success_subtype_and_carries_status(returncode: int) -> None:
    result = parse_claude_cli_output(
        stdout=MEASURED_ORG_DISABLED_403, stderr='', returncode=returncode
    )
    assert result.success is False
    assert result.api_error_status == 403
    assert result.output == ORG_DISABLED_TEXT


def test_budget_cap_subtype_is_not_success() -> None:
    stdout = json.dumps({'subtype': BUDGET_EXHAUSTED_SUBTYPE, 'total_cost_usd': 0.053})
    result = parse_claude_cli_output(stdout=stdout, stderr='', returncode=1)
    assert result.subtype == BUDGET_EXHAUSTED_SUBTYPE
    assert result.success is False


def test_empty_stdout_is_empty_output_and_keeps_stderr() -> None:
    stderr = 'some diagnostic\nsecond line'
    result = parse_claude_cli_output(stdout='', stderr=stderr, returncode=1)
    assert result.success is False
    assert result.subtype == 'error_empty_output'
    assert result.stderr == stderr


@pytest.mark.parametrize('returncode', [0, 1])
def test_non_json_stdout_is_text_output(returncode: int) -> None:
    result = parse_claude_cli_output(stdout='ok', stderr='', returncode=returncode)
    assert result.subtype == 'text_output'
    assert result.output == 'ok'
    assert result.success is (returncode == 0)


def test_is_error_with_structured_output_is_salvaged() -> None:
    stdout = json.dumps({
        'type': 'result',
        'subtype': 'success',
        'is_error': True,
        'structured_output': {'x': 1},
        'result': 'err',
    })
    result = parse_claude_cli_output(stdout=stdout, stderr='', returncode=0)
    assert result.success is True
    assert result.schema_salvaged is True


@pytest.mark.parametrize('stdout', ['42', '[1]', '"ok"', 'null'])
def test_non_object_json_reads_as_text_output(stdout: str) -> None:
    result = parse_claude_cli_output(stdout=stdout, stderr='', returncode=0)
    assert result.subtype == 'text_output'
    assert result.output == stdout


@pytest.mark.parametrize('stdout', ['{"usage": [1]}', '{"usage": "x"}'])
def test_malformed_usage_reads_as_absent_token_counts(stdout: str) -> None:
    result = parse_claude_cli_output(stdout=stdout, stderr='', returncode=0)
    assert result.input_tokens is None
    assert result.output_tokens is None


@pytest.mark.parametrize(
    'stdout',
    [
        '{"messages": [1]}',
        '{"messages": [{"type": "assistant", "content": 5}]}',
        '{"messages": [{"type": "assistant", "content": [{"type": "text"}]}]}',
        '{"result": ["x"]}',
        '{"result": 5}',
    ],
)
def test_malformed_output_fields_read_as_empty_output(stdout: str) -> None:
    result = parse_claude_cli_output(stdout=stdout, stderr='', returncode=0)
    assert result.output == ''


@pytest.mark.parametrize(
    'stdout',
    ['{"api_error_status": "403"}', '{"api_error_status": [1]}', '{"api_error_status": true}'],
)
def test_non_int_api_error_status_reads_as_absent(stdout: str) -> None:
    result = parse_claude_cli_output(stdout=stdout, stderr='', returncode=0)
    assert result.api_error_status is None


def test_assistant_messages_fallback_joins_text_blocks() -> None:
    stdout = json.dumps({
        'subtype': 'success',
        'messages': [
            {'type': 'assistant', 'content': [{'type': 'text', 'text': 'hi'}, 'there']},
        ],
    })
    result = parse_claude_cli_output(stdout=stdout, stderr='', returncode=0)
    assert result.output == 'hi\nthere'
