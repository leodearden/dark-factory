"""Tests for _parse_codex_output and _parse_gemini_output parsers in invoke.py."""

from __future__ import annotations

import json

from shared.cli_invoke import _SubprocessResult

from orchestrator.agents.invoke import _parse_codex_output, _parse_gemini_output


def _make_subprocess_result(
    stdout: str = '',
    stderr: str = '',
    returncode: int = 0,
    duration_ms: int = 100,
) -> _SubprocessResult:
    """Construct a _SubprocessResult with sensible defaults."""
    return _SubprocessResult(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        duration_ms=duration_ms,
    )


class TestParseGeminiNullStats:

    def test_null_stats_does_not_raise(self):
        """_parse_gemini_output does not raise AttributeError when stats is JSON null."""
        payload = json.dumps({'response': 'ok', 'stats': None})
        result = _make_subprocess_result(stdout=payload)
        # Must not raise AttributeError
        agent_result = _parse_gemini_output(result, 'gemini-3-flash')
        assert agent_result.success is True
        assert agent_result.output == 'ok'
        assert agent_result.cost_usd == 0.0


class TestParseGeminiValidStats:

    def test_valid_stats_computes_nonzero_cost(self):
        """_parse_gemini_output computes non-zero cost when stats contains token counts."""
        payload = json.dumps({
            'response': 'hello',
            'stats': {'input_tokens': 100, 'output_tokens': 50},
        })
        result = _make_subprocess_result(stdout=payload)
        agent_result = _parse_gemini_output(result, 'gemini-3-flash')
        assert agent_result.success is True
        assert agent_result.output == 'hello'
        # gemini-3-flash: input=0.075/1M, output=0.30/1M
        # cost = (100 * 0.075 + 50 * 0.30) / 1_000_000 = 0.0000225
        assert agent_result.cost_usd > 0.0


class TestParseCodexNullUsage:

    def test_null_usage_does_not_raise(self):
        """_parse_codex_output does not raise AttributeError when usage is JSON null."""
        events = [
            {'type': 'thread.started', 'thread_id': 'tid-1'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'hello'}},
            {'type': 'turn.completed', 'usage': None},
        ]
        payload = '\n'.join(json.dumps(e) for e in events)
        result = _make_subprocess_result(stdout=payload)
        # Must not raise AttributeError
        agent_result = _parse_codex_output(result, 'o4-mini')
        assert agent_result.success is True
        assert agent_result.output == 'hello'
        assert agent_result.cost_usd == 0.0
        assert agent_result.turns == 1


class TestParseCodexValidUsage:

    def test_valid_usage_computes_nonzero_cost_and_turns(self):
        """_parse_codex_output computes non-zero cost and turns == 1 for real usage."""
        events = [
            {'type': 'thread.started', 'thread_id': 'tid-2'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'done'}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 200, 'output_tokens': 100}},
        ]
        payload = '\n'.join(json.dumps(e) for e in events)
        result = _make_subprocess_result(stdout=payload)
        agent_result = _parse_codex_output(result, 'o4-mini')
        assert agent_result.success is True
        assert agent_result.turns == 1
        # o4-mini: input=1.10/1M, output=4.40/1M
        # cost = (200 * 1.10 + 100 * 4.40) / 1_000_000 = 0.00066
        assert agent_result.cost_usd > 0.0
