"""Tests for reviewer trial corpus data model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.evals.reviewer_trial.corpus import (
    CorpusDiff,
    CorpusManifest,
    GroundTruthIssue,
)


def _make_gt(id: str = 'gt_1', severity: str = 'blocking') -> GroundTruthIssue:
    return GroundTruthIssue(
        id=id,
        location='src/foo.py:42',
        category='test_category',
        severity=severity,
        description='Test issue',
        mutation_type='test_mutation',
    )


def _make_diff(
    diff_id: str = 'test_diff',
    language: str = 'python',
    source: str = 'synthetic',
    ground_truth: list[GroundTruthIssue] | None = None,
) -> CorpusDiff:
    return CorpusDiff(
        diff_id=diff_id,
        language=language,
        source=source,
        diff_text='--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new',
        description='Test diff',
        ground_truth=ground_truth or [_make_gt()],
    )


class TestGroundTruthIssue:
    def test_fields(self) -> None:
        gt = _make_gt()
        assert gt.id == 'gt_1'
        assert gt.severity == 'blocking'
        assert gt.category == 'test_category'


class TestCorpusDiff:
    def test_blocking_issues(self) -> None:
        diff = _make_diff(ground_truth=[
            _make_gt('b1', 'blocking'),
            _make_gt('s1', 'suggestion'),
            _make_gt('b2', 'blocking'),
        ])
        blocking = diff.blocking_issues()
        assert len(blocking) == 2
        assert all(gt.severity == 'blocking' for gt in blocking)

    def test_suggestion_issues(self) -> None:
        diff = _make_diff(ground_truth=[
            _make_gt('b1', 'blocking'),
            _make_gt('s1', 'suggestion'),
        ])
        suggestions = diff.suggestion_issues()
        assert len(suggestions) == 1
        assert suggestions[0].id == 's1'


class TestCorpusManifest:
    def test_filter_by_language(self) -> None:
        manifest = CorpusManifest(diffs=[
            _make_diff('py1', language='python'),
            _make_diff('rs1', language='rust'),
            _make_diff('py2', language='python'),
        ])
        py = manifest.filter_by_language('python')
        assert len(py) == 2
        assert all(d.language == 'python' for d in py)

    def test_filter_by_source(self) -> None:
        manifest = CorpusManifest(diffs=[
            _make_diff('s1', source='synthetic'),
            _make_diff('r1', source='real_world'),
        ])
        synthetic = manifest.filter_by_source('synthetic')
        assert len(synthetic) == 1
        assert synthetic[0].diff_id == 's1'

    def test_get_diff(self) -> None:
        manifest = CorpusManifest(diffs=[_make_diff('d1'), _make_diff('d2')])
        assert manifest.get_diff('d1') is not None
        assert manifest.get_diff('d1').diff_id == 'd1'
        assert manifest.get_diff('nonexistent') is None

    def test_save_and_load(self, tmp_path: Path) -> None:
        gt = _make_gt()
        diff = _make_diff(ground_truth=[gt])
        manifest = CorpusManifest(diffs=[diff])

        manifest_path = tmp_path / 'manifest.json'
        manifest.save(manifest_path)

        # Check files were created
        assert (tmp_path / 'synthetic' / 'test_diff.diff').exists()
        assert (tmp_path / 'annotations' / 'test_diff.json').exists()
        assert manifest_path.exists()

        # Reload
        loaded = CorpusManifest.load(manifest_path)
        assert len(loaded.diffs) == 1
        assert loaded.diffs[0].diff_id == 'test_diff'
        assert loaded.diffs[0].language == 'python'
        assert loaded.diffs[0].diff_text == diff.diff_text
        assert len(loaded.diffs[0].ground_truth) == 1
        assert loaded.diffs[0].ground_truth[0].id == 'gt_1'

    def test_load_real_manifest(self) -> None:
        """Verify the actual trial corpus manifest loads without error."""
        manifest_path = (
            Path(__file__).parent.parent
            / 'src' / 'orchestrator' / 'evals' / 'reviewer_trial'
            / 'corpus' / 'manifest.json'
        )
        if not manifest_path.exists():
            pytest.skip('Corpus manifest not found')

        manifest = CorpusManifest.load(manifest_path)
        assert len(manifest.diffs) == 15
        assert all(d.diff_text for d in manifest.diffs)
        assert all(d.ground_truth for d in manifest.diffs)

        # Check language distribution
        py = manifest.filter_by_language('python')
        rs = manifest.filter_by_language('rust')
        ts = manifest.filter_by_language('typescript')
        assert len(py) == 6
        assert len(rs) == 6
        assert len(ts) == 3
