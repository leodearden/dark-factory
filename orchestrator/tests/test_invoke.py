"""Tests for orchestrator agent invocation parsers (codex, gemini)."""

from __future__ import annotations

import json

from shared.cli_invoke import AgentResult, _SubprocessResult

from orchestrator.agents.invoke import _parse_codex_output, _parse_gemini_output


def _subprocess_result(stdout: str, returncode: int = 0) -> _SubprocessResult:
    return _SubprocessResult(
        stdout=stdout,
        stderr='',
        returncode=returncode,
        duration_ms=50,
    )


class TestParseCodexTokens:
    """_parse_codex_output populates input_tokens and output_tokens."""

    def test_parses_token_fields_from_turn_completed(self):
        """Token counts are summed from turn.completed usage blocks."""
        events = [
            {'type': 'thread.started', 'thread_id': 'thread-1'},
            {'type': 'turn.started'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'hello'}},
            {
                'type': 'turn.completed',
                'usage': {'input_tokens': 2000, 'output_tokens': 600},
            },
        ]
        stdout = '\n'.join(json.dumps(e) for e in events)
        result = _parse_codex_output(_subprocess_result(stdout), 'gpt-5.4')
        assert result.input_tokens == 2000
        assert result.output_tokens == 600
        assert result.cache_read_tokens is None
        assert result.cache_create_tokens is None

    def test_accumulates_tokens_across_multiple_turns(self):
        """Token counts are summed across all turn.completed events."""
        events = [
            {'type': 'turn.completed', 'usage': {'input_tokens': 1000, 'output_tokens': 300}},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'more'}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 800, 'output_tokens': 250}},
        ]
        stdout = '\n'.join(json.dumps(e) for e in events)
        result = _parse_codex_output(_subprocess_result(stdout), 'gpt-5.4')
        assert result.input_tokens == 1800
        assert result.output_tokens == 550

    def test_empty_output_gives_none_tokens(self):
        """Empty output returns None for token fields (not zero)."""
        result = _parse_codex_output(_subprocess_result(''), 'gpt-5.4')
        assert result.input_tokens is None
        assert result.output_tokens is None

    def test_no_usage_events_gives_none_tokens(self):
        """JSONL with no turn.completed events returns None for token fields."""
        events = [
            {'type': 'thread.started', 'thread_id': 'tid'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'hi'}},
        ]
        stdout = '\n'.join(json.dumps(e) for e in events)
        result = _parse_codex_output(_subprocess_result(stdout), 'gpt-5.4')
        assert result.input_tokens is None
        assert result.output_tokens is None


class TestParseGeminiTokens:
    """_parse_gemini_output populates input_tokens and output_tokens."""

    def test_parses_token_fields_from_stats(self):
        """Token counts are extracted from the stats block."""
        data = {
            'response': 'hello from gemini',
            'stats': {
                'input_tokens': 3000,
                'output_tokens': 1200,
                'turns': 1,
            },
        }
        result = _parse_gemini_output(_subprocess_result(json.dumps(data)), 'gemini-3.1-pro-preview')
        assert result.input_tokens == 3000
        assert result.output_tokens == 1200
        assert result.cache_read_tokens is None
        assert result.cache_create_tokens is None

    def test_zero_tokens_gives_none(self):
        """Stats with 0 tokens converts to None (unavailable, not zero tokens used)."""
        data = {
            'response': 'hi',
            'stats': {'input_tokens': 0, 'output_tokens': 0, 'turns': 1},
        }
        result = _parse_gemini_output(_subprocess_result(json.dumps(data)), 'gemini-3-flash')
        assert result.input_tokens is None
        assert result.output_tokens is None

    def test_empty_output_gives_none_tokens(self):
        """Empty output returns None for token fields."""
        result = _parse_gemini_output(_subprocess_result(''), 'gemini-3-flash')
        assert result.input_tokens is None
        assert result.output_tokens is None

    def test_no_stats_gives_none_tokens(self):
        """Missing stats key gives None for token fields."""
        data = {'response': 'hi'}
        result = _parse_gemini_output(_subprocess_result(json.dumps(data)), 'gemini-3-flash')
        assert result.input_tokens is None
        assert result.output_tokens is None
