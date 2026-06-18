"""Tests for shared.agent_result — extract_agent_verdict distinguishable ERROR sentinel."""

from __future__ import annotations

import logging

from shared.agent_result import AgentVerdict, extract_agent_verdict


class TestExtractAgentVerdictWarningKey:
    """Failure path: agent_loop returns {'warning': ...} shape — must produce loud sentinel."""

    def test_warning_key_produces_agent_failed_sentinel(self, caplog):
        """PRD-signal test: {'warning': 'no_tool_calls'} → agent-failed:no_tool_calls + WARNING."""
        with caplog.at_level(logging.WARNING):
            result = extract_agent_verdict(
                {'warning': 'no_tool_calls', 'text': 'agent gave up'},
                default_verdict='ERROR',
                error_summary='verify_failed',
            )

        assert isinstance(result, AgentVerdict)
        assert result.summary == 'agent-failed:no_tool_calls'
        assert result.verdict == 'ERROR'
        assert result.failed is True

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'expected at least one WARNING log record on failure'
        assert any('no_tool_calls' in r.message for r in warning_records), (
            f'no WARNING record contained token "no_tool_calls"; records: {[r.message for r in warning_records]}'
        )


class TestExtractAgentVerdictHappyPath:
    """Success path: dict with truthy 'verdict' — must return real verdict silently."""

    def test_happy_path_returns_real_verdict_silently(self, caplog):
        """{'verdict': 'confirmed', ...} → AgentVerdict with real verdict, no WARNING."""
        payload = {'verdict': 'confirmed', 'summary': 'looks good', 'confidence': 0.9}
        with caplog.at_level(logging.WARNING):
            result = extract_agent_verdict(
                payload,
                default_verdict='ERROR',
                error_summary='unused',
            )

        assert isinstance(result, AgentVerdict)
        assert result.verdict == 'confirmed'
        assert result.summary == 'looks good'
        assert result.failed is False
        assert result.raw == payload, 'raw must pass the full input dict through'

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            f'success path must emit NO WARNING; got: {[r.message for r in warning_records]}'
        )


class TestExtractAgentVerdictFallback:
    """Fallback path: None / non-dict / dict-without-warning all use error_summary as token."""

    def test_none_result_uses_error_summary_token(self, caplog):
        """None input → agent-failed:<error_summary>, failed=True, WARNING emitted."""
        with caplog.at_level(logging.WARNING):
            result = extract_agent_verdict(
                None,
                default_verdict='inconclusive',
                error_summary='unparseable_output',
            )

        assert isinstance(result, AgentVerdict)
        assert result.summary == 'agent-failed:unparseable_output'
        assert result.verdict == 'inconclusive'
        assert result.failed is True

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'expected at least one WARNING log record on None input'

    def test_dict_without_verdict_or_warning_uses_error_summary_token(self, caplog):
        """Dict with neither 'verdict' nor 'warning' → agent-failed:<error_summary>."""
        with caplog.at_level(logging.WARNING):
            result = extract_agent_verdict(
                {'text': 'oops'},
                default_verdict='inconclusive',
                error_summary='unparseable_output',
            )

        assert isinstance(result, AgentVerdict)
        assert result.summary == 'agent-failed:unparseable_output'
        assert result.verdict == 'inconclusive'
        assert result.failed is True

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'expected at least one WARNING log record on warning-less dict'
