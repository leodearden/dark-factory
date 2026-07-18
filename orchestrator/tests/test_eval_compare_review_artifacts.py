"""Tests for evals/compare.py review-artifact reading.

Covers the post-2484-migration seam: the reviewer's verdict/issues payload
for an eval run now lives in the MCP verdict envelope
(``.task/verdicts/reviewer_comprehensive.json``), not only the legacy
``.task/reviews/`` directory. ``_read_review_artifacts`` must prefer the
verdict envelope and fall back to legacy reviews/ for pre-migration runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.evals.compare import _read_reviewer_verdict_envelopes


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    wt = tmp_path / 'worktree'
    wt.mkdir()
    return wt


@pytest.fixture
def artifacts(worktree: Path) -> TaskArtifacts:
    ta = TaskArtifacts(worktree)
    ta.init('task-1', 'Test Task', 'A test task description')
    return ta


def _envelope(role: str, payload: dict) -> dict:
    return {
        'role': role,
        'schema_version': 1,
        'session_id': 's',
        'emitted_at': 't',
        'verdict': payload,
    }


# ---------------------------------------------------------------------------
# Cycle A: pure helper _read_reviewer_verdict_envelopes
# ---------------------------------------------------------------------------

class TestReadReviewerVerdictEnvelopes:
    def test_unwraps_envelope_verdict(self, artifacts: TaskArtifacts):
        payload = {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'ISSUES_FOUND',
            'issues': [
                {
                    'severity': 'blocking',
                    'category': 'correctness',
                    'location': 'x.py:1',
                    'description': 'd',
                },
            ],
            'summary': 'ENV',
        }
        artifacts.write_verdict(
            'reviewer_comprehensive', _envelope('reviewer_comprehensive', payload),
        )

        result = _read_reviewer_verdict_envelopes(artifacts)

        assert result == {'reviewer_comprehensive': payload}

    def test_excludes_non_reviewer_envelopes(self, artifacts: TaskArtifacts):
        payload = {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'ENV',
        }
        artifacts.write_verdict(
            'reviewer_comprehensive', _envelope('reviewer_comprehensive', payload),
        )
        artifacts.write_verdict('judge', _envelope('judge', {'complete': True}))

        result = _read_reviewer_verdict_envelopes(artifacts)

        assert set(result) == {'reviewer_comprehensive'}

    def test_skips_corrupt_envelope(self, artifacts: TaskArtifacts):
        (artifacts.root / 'verdicts' / 'reviewer_bad.json').write_text('{not valid json')

        result = _read_reviewer_verdict_envelopes(artifacts)

        assert result == {}

    def test_empty_when_verdicts_dir_has_no_reviewer_files(self, artifacts: TaskArtifacts):
        artifacts.write_verdict('judge', _envelope('judge', {'complete': True}))

        result = _read_reviewer_verdict_envelopes(artifacts)

        assert result == {}

    def test_empty_when_verdicts_dir_absent(self, worktree: Path):
        # No init() call — .task/verdicts/ never created.
        artifacts = TaskArtifacts(worktree)

        result = _read_reviewer_verdict_envelopes(artifacts)

        assert result == {}
