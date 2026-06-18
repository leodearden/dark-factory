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
        """None input → agent-failed:<error_summary>, failed=True, WARNING emitted, raw=None."""
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
        assert result.raw is None, 'raw must be None when input is not a dict'

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'expected at least one WARNING log record on None input'

    def test_dict_without_verdict_or_warning_uses_error_summary_token(self, caplog):
        """Dict with neither 'verdict' nor 'warning' → agent-failed:<error_summary>, raw passes through."""
        payload = {'text': 'oops'}
        with caplog.at_level(logging.WARNING):
            result = extract_agent_verdict(
                payload,
                default_verdict='inconclusive',
                error_summary='unparseable_output',
            )

        assert isinstance(result, AgentVerdict)
        assert result.summary == 'agent-failed:unparseable_output'
        assert result.verdict == 'inconclusive'
        assert result.failed is True
        assert result.raw == payload, 'raw must pass the dict through even on the failure path'

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'expected at least one WARNING log record on warning-less dict'


class TestExtractAgentVerdictEdgeCases:
    """Edge cases that pin the truthiness boundary and robustness guards."""

    def test_verdict_present_without_summary_defaults_to_empty_string(self, caplog):
        """{'verdict': 'x'} with no 'summary' key → summary defaults to '' silently."""
        with caplog.at_level(logging.WARNING):
            result = extract_agent_verdict(
                {'verdict': 'x'},
                default_verdict='ERROR',
                error_summary='unused',
            )

        assert result.verdict == 'x'
        assert result.summary == '', "missing 'summary' key must default to empty string"
        assert result.failed is False
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], 'success path must stay silent even when summary is absent'

    def test_empty_string_verdict_routes_to_failure_sentinel(self, caplog):
        """{'verdict': ''} — falsy verdict — routes to the failure path (not success)."""
        with caplog.at_level(logging.WARNING):
            result = extract_agent_verdict(
                {'verdict': ''},
                default_verdict='inconclusive',
                error_summary='empty_verdict',
            )

        assert result.verdict == 'inconclusive'
        assert result.summary == 'agent-failed:empty_verdict'
        assert result.failed is True
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'falsy verdict must trigger a WARNING (not a silent success)'

    def test_tuple_misuse_emits_distinct_warning(self, caplog):
        """2-tuple from AgentLoop.run() → a distinct 'unpack' WARNING, failure result.

        A caller who forgets to unpack (result_dict, journal) and passes the whole
        tuple must get a loud, actionable message rather than a silent failure with
        an unhelpful error_summary token.
        """
        with caplog.at_level(logging.WARNING):
            result = extract_agent_verdict(
                ({'verdict': 'confirmed'}, ['journal_entry']),
                default_verdict='inconclusive',
                error_summary='forgot_to_unpack',
            )

        assert result.failed is True
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'expected WARNING on 2-tuple input'
        assert any(
            'tuple' in r.message.lower() or 'unpack' in r.message.lower()
            for r in warning_records
        ), (
            'WARNING must mention the tuple/unpack so the caller knows how to fix it; '
            f'got: {[r.message for r in warning_records]}'
        )

    def test_truthy_verdict_takes_precedence_over_warning_key(self, caplog):
        """When both 'verdict' (truthy) and 'warning' are present, verdict wins silently.

        Documents intentional behavior: the success gate is checked first and
        short-circuits, so any co-present 'warning' key is intentionally dropped.
        Callers that need to see the warning can inspect result.raw.
        """
        payload = {'verdict': 'confirmed', 'warning': 'partial_result', 'summary': 'ok'}
        with caplog.at_level(logging.WARNING):
            result = extract_agent_verdict(
                payload,
                default_verdict='ERROR',
                error_summary='unused',
            )

        assert result.verdict == 'confirmed'
        assert result.failed is False
        assert result.raw == payload, 'raw passes the full payload through'
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            "verdict takes precedence; no WARNING should fire for this payload; "
            f"got: {[r.message for r in warning_records]}"
        )
