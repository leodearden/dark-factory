"""Tests for ReviewCheckpoint handling of AllAccountsCappedException.

Before the explicit handler is added in step-6, AllAccountsCappedException
propagates out of _run_review() through run_focused(), crashing the review
asyncio.Task with no graceful downgrade.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from shared.cli_invoke import AllAccountsCappedException

from orchestrator.review_checkpoint import ReviewCheckpoint, ReviewReport
from orchestrator.verify import VerifyResult


def _make_checkpoint() -> ReviewCheckpoint:
    """Construct a minimal ReviewCheckpoint with mocked config."""
    config = MagicMock()
    # Use /tmp as project_root — avoids the '/tmp/pytest-' guard in _run_review
    # without needing the directory to actually exist (since run_full_verification
    # is mocked out in these tests).
    config.project_root = Path('/tmp')
    # Ensure string-attr access used by _run_review returns sensible values
    config.models.deep_reviewer = 'opus'
    config.max_turns.deep_reviewer = 50
    config.budgets.deep_reviewer = 10.0
    config.effort.deep_reviewer = 'max'
    config.backends.deep_reviewer = 'claude'
    config.fused_memory.project_id = 'test-project'
    config.escalation.host = 'localhost'
    config.escalation.port = 9999

    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {'mcpServers': {}}

    cp = ReviewCheckpoint(config, mcp=mcp, usage_gate=None)
    return cp


_PHASE1_RESULT = VerifyResult(
    passed=True, summary='OK', test_output='', lint_output='', type_output=''
)


@pytest.mark.asyncio
class TestReviewCheckpointCapHandling:
    """Verify ReviewCheckpoint._run_review handles AllAccountsCappedException gracefully."""

    async def test_run_review_handles_all_accounts_capped(
        self, monkeypatch, caplog
    ):
        """run_focused must return an empty ReviewReport, not raise, on cap exhaustion.

        Before step-6 impl: AllAccountsCappedException propagates out of
        _run_review() → run_focused() → test assertion fails with exception.
        After step-6 impl: exception is caught, empty ReviewReport returned.
        """
        checkpoint = _make_checkpoint()

        cap_exc = AllAccountsCappedException(
            retries=4, elapsed_secs=300.0, label='Review checkpoint [x]'
        )

        monkeypatch.setattr(
            'orchestrator.review_checkpoint.invoke_with_cap_retry',
            AsyncMock(side_effect=cap_exc),
        )
        monkeypatch.setattr(
            'orchestrator.review_checkpoint.run_full_verification',
            AsyncMock(return_value=_PHASE1_RESULT),
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.review_checkpoint'):
            report = await checkpoint.run_focused()

        # Must return a ReviewReport (not raise)
        assert isinstance(report, ReviewReport)
        assert report.mode == 'focused'
        assert report.findings_count == 0
        assert report.tasks_created == []
        assert report.cost_usd == 0.0

        # Must emit a warning containing 'all accounts capped'
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            'all accounts capped' in t.lower() for t in warning_texts
        ), f'Expected warning with "all accounts capped", got: {warning_texts}'
