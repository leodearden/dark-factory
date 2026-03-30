"""Tests for _parse_codex_output and _parse_gemini_output parsers in invoke.py."""

from __future__ import annotations

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
