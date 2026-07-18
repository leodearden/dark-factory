"""Tests for evals/compare.py review-artifact reading.

Covers the post-2484-migration seam: the reviewer's verdict/issues payload
for an eval run now lives in the MCP verdict envelope
(``.task/verdicts/reviewer_comprehensive.json``), not only the legacy
``.task/reviews/`` directory. ``_read_review_artifacts`` must prefer the
verdict envelope and fall back to legacy reviews/ for pre-migration runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.evals.compare import (
    _format_review_artifacts,
    _read_legacy_reviews,
    _read_review_artifacts,
    _read_reviewer_verdict_envelopes,
)


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

    def test_skips_malformed_role_filename(self, artifacts: TaskArtifacts):
        # A well-formed envelope for the real reviewer role...
        good_payload = {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'ENV',
        }
        artifacts.write_verdict(
            'reviewer_comprehensive', _envelope('reviewer_comprehensive', good_payload),
        )
        # ...alongside a stray file whose stem ("reviewer.stray") contains a
        # '.' and so is not a valid verdict role (TaskArtifacts.read_verdict
        # raises ValueError for it via _validate_verdict_role). It matches
        # the reviewer*.json glob, so it must be skipped-with-warning rather
        # than crashing the whole read.
        (artifacts.root / 'verdicts' / 'reviewer.stray.json').write_text(
            json.dumps(_envelope('reviewer.stray', {'reviewer': 'x', 'verdict': 'PASS',
                                                      'issues': [], 'summary': 's'})),
        )

        result = _read_reviewer_verdict_envelopes(artifacts)

        assert result == {'reviewer_comprehensive': good_payload}

    def test_empty_when_verdicts_dir_has_no_reviewer_files(self, artifacts: TaskArtifacts):
        artifacts.write_verdict('judge', _envelope('judge', {'complete': True}))

        result = _read_reviewer_verdict_envelopes(artifacts)

        assert result == {}

    def test_empty_when_verdicts_dir_absent(self, worktree: Path):
        # No init() call — .task/verdicts/ never created.
        artifacts = TaskArtifacts(worktree)

        result = _read_reviewer_verdict_envelopes(artifacts)

        assert result == {}


# ---------------------------------------------------------------------------
# Cycle A2: pure helper _read_legacy_reviews (corrupt-file tolerance)
# ---------------------------------------------------------------------------

class TestReadLegacyReviews:
    def test_keyed_by_reviewer_field(self, artifacts: TaskArtifacts):
        legacy = {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'LEGACY',
        }
        artifacts.write_review('reviewer_comprehensive', legacy)

        result = _read_legacy_reviews(artifacts)

        assert result == {'reviewer_comprehensive': legacy}

    def test_skips_corrupt_file_keeps_good_ones(self, artifacts: TaskArtifacts):
        good = {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'LEGACY-GOOD',
        }
        artifacts.write_review('reviewer_comprehensive', good)
        (artifacts.root / 'reviews' / 'reviewer_bad.json').write_text('{not valid json')

        result = _read_legacy_reviews(artifacts)

        assert result == {'reviewer_comprehensive': good}

    def test_empty_when_reviews_dir_absent(self, worktree: Path):
        # No init() call — .task/reviews/ never created.
        artifacts = TaskArtifacts(worktree)

        result = _read_legacy_reviews(artifacts)

        assert result == {}


# ---------------------------------------------------------------------------
# Cycle B: integrated _read_review_artifacts across both layouts
# ---------------------------------------------------------------------------

class TestReadReviewArtifacts:
    def test_envelope_present_only(self, worktree: Path):
        ta = TaskArtifacts(worktree)
        ta.init('task-1', 'Test Task', 'desc')
        payload = {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'ISSUES_FOUND',
            'issues': [],
            'summary': 'ENV',
        }
        ta.write_verdict('reviewer_comprehensive', _envelope('reviewer_comprehensive', payload))

        result = _read_review_artifacts(str(worktree))

        assert result == {'reviewer_comprehensive': payload}
        formatted = _format_review_artifacts(result)
        assert 'ENV' in formatted
        assert '(no review data available)' not in formatted

    def test_legacy_reviews_only(self, worktree: Path):
        ta = TaskArtifacts(worktree)
        ta.init('task-1', 'Test Task', 'desc')
        legacy = {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'LEGACY',
        }
        ta.write_review('reviewer_comprehensive', legacy)

        result = _read_review_artifacts(str(worktree))

        assert result == {'reviewer_comprehensive': legacy}
        formatted = _format_review_artifacts(result)
        assert 'LEGACY' in formatted
        assert '(no review data available)' not in formatted

    def test_legacy_review_corrupt_file_skipped(self, worktree: Path):
        # A corrupt/unreadable file in a pre-migration reviews/ dir must be
        # skipped (with a warning), not raised — restoring parity with the
        # pre-envelope reader, which tolerated per-file corruption.
        ta = TaskArtifacts(worktree)
        ta.init('task-1', 'Test Task', 'desc')
        good = {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'LEGACY-GOOD',
        }
        ta.write_review('reviewer_comprehensive', good)
        (ta.root / 'reviews' / 'reviewer_bad.json').write_text('{not valid json')

        result = _read_review_artifacts(str(worktree))

        assert result == {'reviewer_comprehensive': good}
        formatted = _format_review_artifacts(result)
        assert 'LEGACY-GOOD' in formatted

    def test_envelope_preferred_over_legacy(self, worktree: Path):
        ta = TaskArtifacts(worktree)
        ta.init('task-1', 'Test Task', 'desc')
        env_payload = {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'ISSUES_FOUND',
            'issues': [],
            'summary': 'ENV',
        }
        legacy_payload = {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'LEGACY',
        }
        ta.write_verdict('reviewer_comprehensive', _envelope('reviewer_comprehensive', env_payload))
        ta.write_review('reviewer_comprehensive', legacy_payload)

        result = _read_review_artifacts(str(worktree))

        assert result == {'reviewer_comprehensive': env_payload}

    def test_neither_present_returns_empty(self, worktree: Path):
        # worktree exists but .task/ was never initialized.
        result = _read_review_artifacts(str(worktree))

        assert result == {}
        assert _format_review_artifacts(result) == '(no review data available)'

    def test_neither_present_missing_worktree(self, tmp_path: Path):
        missing = tmp_path / 'does-not-exist'

        result = _read_review_artifacts(str(missing))

        assert result == {}
