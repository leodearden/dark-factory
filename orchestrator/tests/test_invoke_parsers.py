"""Tests for _parse_codex_output and _parse_gemini_output parsers in invoke.py."""

from __future__ import annotations

import json

from shared.cli_invoke import _SubprocessResult
from orchestrator.agents.invoke import _parse_codex_output, _parse_gemini_output


def _make_subprocess_result(**overrides) -> _SubprocessResult:
    """Construct a _SubprocessResult with sensible defaults."""
    defaults = dict(
        stdout='',
        stderr='',
        returncode=0,
        duration_ms=100,
    )
    defaults.update(overrides)
    return _SubprocessResult(**defaults)


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
