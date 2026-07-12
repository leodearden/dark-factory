"""Tests for the verdict-tools MCP server."""

from __future__ import annotations

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp.verdict_tools import (
    _submit_review_verdict,
)


@pytest.fixture()
def artifacts(tmp_path):
    """TaskArtifacts pointing at a temporary worktree."""
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


# ---------------------------------------------------------------------------
# Standalone _impl tests (independent of the MCP tool/pydantic boundary)
# ---------------------------------------------------------------------------


class TestSubmitReviewVerdict:
    def test_writes_envelope_and_returns_ok(self, artifacts):
        result = _submit_review_verdict(
            artifacts,
            role='test_analyst',
            session_id='sess-1',
            reviewer='test_analyst',
            verdict='PASS',
            issues=[],
            summary='ok',
        )
        assert result == {'status': 'ok', 'role': 'test_analyst'}

        envelope = artifacts.read_verdict('test_analyst')
        assert envelope is not None
        assert envelope['role'] == 'test_analyst'
        assert envelope['schema_version'] == 1
        assert envelope['session_id'] == 'sess-1'
        assert isinstance(envelope['emitted_at'], str) and envelope['emitted_at']
        assert envelope['verdict'] == {
            'reviewer': 'test_analyst',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'ok',
        }
