"""Corpus-integrity gate for the expanded reviewer_trial corpus (task 2495).

Loads the REAL, committed corpus manifest + adjudication log and asserts
the audit_corpus signals hold over it: >=50 diffs, every diff has a split
approximating the 2:1:7 ratio, every mined diff carries provenance, the
adjudication log covers every diff_id, and a documented spot-check subset
exists. Also asserts the scorer runs green over the expanded corpus --
verified structurally/deterministically (no LLM call) via the empty-reviews
early-return path in ``scorer.match_issues``, per the design decision in
plan.json ("Scorer runs green" section) that avoids LLM nondeterminism in
this gate. The full-LLM ``corpus-sanity`` CLI command remains the operator
tool for actually exercising the reviewer panel + haiku matcher.

This is the RED half of step-17/step-18: it fails against the pre-expansion
15-diff corpus and only passes once the offline `mine` generation run
(step-18) has expanded corpus/ to >=50 diffs with splits + provenance +
an adjudication log.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.evals.reviewer_trial.adjudication import AdjudicationLog
from orchestrator.evals.reviewer_trial.corpus import CorpusManifest
from orchestrator.evals.reviewer_trial.mining import audit_corpus
from orchestrator.evals.reviewer_trial.runner import PanelRunResult
from orchestrator.evals.reviewer_trial.scorer import score_panel_run

_PKG_DIR = (
    Path(__file__).parent.parent
    / 'src' / 'orchestrator' / 'evals' / 'reviewer_trial'
)
_MANIFEST_PATH = _PKG_DIR / 'corpus' / 'manifest.json'
_ADJUDICATION_LOG_PATH = _PKG_DIR / 'corpus' / 'adjudication_log.jsonl'


def _load_real_manifest() -> CorpusManifest:
    if not _MANIFEST_PATH.exists():
        pytest.skip('Corpus manifest not found')
    return CorpusManifest.load(_MANIFEST_PATH)


class TestCorpusIntegrity:
    """The committed corpus (post task-2495 expansion) satisfies every
    audit_corpus signal."""

    def test_audit_corpus_ok_over_real_corpus(self) -> None:
        manifest = _load_real_manifest()
        log = AdjudicationLog.load(_ADJUDICATION_LOG_PATH)

        report = audit_corpus(manifest, log, min_diffs=50)

        assert report.ok is True, f'audit_corpus failures: {report.failures}'

    def test_real_corpus_has_at_least_50_diffs(self) -> None:
        manifest = _load_real_manifest()
        assert len(manifest.diffs) >= 50

    def test_every_diff_has_a_split(self) -> None:
        manifest = _load_real_manifest()
        missing = [d.diff_id for d in manifest.diffs if d.split is None]
        assert not missing, f'diffs missing a split: {missing}'

    def test_adjudication_log_covers_all_diffs(self) -> None:
        manifest = _load_real_manifest()
        log = AdjudicationLog.load(_ADJUDICATION_LOG_PATH)

        coverage = log.coverage(d.diff_id for d in manifest.diffs)

        assert coverage.ok, f'diffs missing an adjudication entry: {coverage.missing}'

    def test_documented_spot_check_subset_is_non_empty(self) -> None:
        log = AdjudicationLog.load(_ADJUDICATION_LOG_PATH)
        assert log.spot_check_subset(), 'adjudication_log has no flagged spot-check subset'


class TestScorerStructuralGreen:
    """Scorer runs green over the expanded corpus -- verified
    structurally/deterministically, not via a live (nondeterministic,
    costly) LLM panel+matcher run. Empty reviews trigger match_issues'
    no-LLM early return, exercising corpus load + dedup + metric math."""

    @pytest.mark.asyncio
    async def test_score_panel_run_over_sampled_real_diff(self) -> None:
        manifest = _load_real_manifest()
        assert manifest.diffs, 'corpus is empty'
        diff = manifest.diffs[0]
        stub_run = PanelRunResult(variant_name='stub', diff_id=diff.diff_id, reviews={})

        result = await score_panel_run(stub_run, diff)

        assert result.diff_id == diff.diff_id
        assert result.matches == []
        assert result.match_cost_usd == 0.0
