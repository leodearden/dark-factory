"""Tests for the prior-round-resolution suppression helpers (task 2523).

Pure sibling of ``amendment_scope.py``:

- :func:`load_prior_round_suggestions` reconstructs the prior-round
  suggestion-severity set from the archived ``reviews-amend-*/`` review JSONs,
  mirroring ``artifacts.aggregate_reviews``' partition (skip ``verdict ==
  'ERROR'`` reviews; keep only ``severity != 'blocking'`` issues).
- :func:`partition_by_decisions` applies an adjudicator's per-index verdict,
  suppressing a current suggestion ONLY on an explicit ``SETTLED`` — every
  other value keeps it (fail-safe toward emit).
- :func:`build_resettled_adjudicator_prompt` builds the batched adjudication
  prompt.

Pure functions — no git invocation, no workflow state, tolerant parsing.
"""

from __future__ import annotations

import json
from pathlib import Path


def _issue(
    severity: str, description: str, location: str = 'src/foo.py:1', **extras
) -> dict:
    """Build a review-issue dict with the fields the helpers read.

    Mirrors the ``_sugg`` helper shape in test_amendment_scope.py /
    test_workflow_amendment.py, but keyed on ``severity`` so a caller can build
    both blocking and suggestion issues.
    """
    d: dict = {
        'severity': severity,
        'location': location,
        'category': 'coverage',
        'description': description,
        'suggested_fix': 'do the thing',
    }
    d.update(extras)
    return d


def _write_review(
    root: Path, amend_round: int, reviewer: str, review: dict
) -> None:
    """Write ``<root>/reviews-amend-<N>/<reviewer>.json`` (mirrors write_review)."""
    d = root / f'reviews-amend-{amend_round}'
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{reviewer}.json').write_text(json.dumps(review))


def _load():
    from orchestrator.review_suggestions.prior_round import (
        load_prior_round_suggestions,
    )
    return load_prior_round_suggestions


class TestLoadPriorRoundSuggestions:
    """``load_prior_round_suggestions(artifacts_root)`` — prior-round set."""

    def test_no_archives_returns_empty(self, tmp_path):
        """(a) No reviews-amend-*/ dirs → []."""
        assert _load()(tmp_path) == []

    def test_collects_non_blocking_from_all_archives(self, tmp_path):
        """(b) Flattens suggestion-severity issues across every archive dir."""
        _write_review(tmp_path, 1, 'reviewer_comprehensive', {
            'verdict': 'PASS',
            'issues': [
                _issue('suggestion', 'sugg-a1'),
                _issue('blocking', 'block-a1'),
            ],
        })
        _write_review(tmp_path, 2, 'reviewer_comprehensive', {
            'verdict': 'PASS',
            'issues': [
                _issue('suggestion', 'sugg-b1'),
            ],
        })
        result = _load()(tmp_path)
        assert sorted(i['description'] for i in result) == ['sugg-a1', 'sugg-b1']
        assert all(i.get('severity') != 'blocking' for i in result)

    def test_excludes_blocking_severity(self, tmp_path):
        """(c) Blocking-severity issues are never collected."""
        _write_review(tmp_path, 1, 'reviewer_comprehensive', {
            'verdict': 'PASS',
            'issues': [_issue('blocking', 'block-only')],
        })
        assert _load()(tmp_path) == []

    def test_skips_error_verdict_review(self, tmp_path):
        """(d) A review whose verdict == 'ERROR' is skipped entirely."""
        _write_review(tmp_path, 1, 'reviewer_comprehensive', {
            'verdict': 'ERROR',
            'issues': [_issue('suggestion', 'should-be-skipped')],
        })
        assert _load()(tmp_path) == []

    def test_tolerates_malformed_json_and_missing_issues(self, tmp_path):
        """(e) Non-JSON files and reviews missing `issues` are skipped, not raised."""
        # Malformed / non-JSON archive file.
        d1 = tmp_path / 'reviews-amend-1'
        d1.mkdir(parents=True)
        (d1 / 'reviewer_comprehensive.json').write_text('{ not valid json ')
        # Review missing the `issues` key.
        _write_review(tmp_path, 2, 'reviewer_comprehensive', {'verdict': 'PASS'})
        # A valid archive so we can confirm parsing continues past the bad ones.
        _write_review(tmp_path, 3, 'reviewer_comprehensive', {
            'verdict': 'PASS',
            'issues': [_issue('suggestion', 'good')],
        })
        result = _load()(tmp_path)
        assert [i['description'] for i in result] == ['good']


class TestPartitionByDecisions:
    """``partition_by_decisions(current, decisions)`` — fail-safe partition."""

    @staticmethod
    def _fn():
        from orchestrator.review_suggestions.prior_round import (
            partition_by_decisions,
        )
        return partition_by_decisions

    def test_settled_is_suppressed(self):
        """(a) A suggestion whose decision == SETTLED is suppressed."""
        from orchestrator.review_suggestions.prior_round import (
            NOT_SETTLED,
            SETTLED,
        )
        s0 = _issue('suggestion', 's0')
        s1 = _issue('suggestion', 's1')
        kept, suppressed = self._fn()([s0, s1], [SETTLED, NOT_SETTLED])
        assert kept == [s1]
        assert suppressed == [s0]

    def test_non_settled_values_keep(self):
        """(b) NOT_SETTLED, INCONCLUSIVE, and unknown strings all keep."""
        from orchestrator.review_suggestions.prior_round import (
            INCONCLUSIVE,
            NOT_SETTLED,
        )
        items = [_issue('suggestion', f's{i}') for i in range(3)]
        kept, suppressed = self._fn()(
            items, [NOT_SETTLED, INCONCLUSIVE, 'garbage_value']
        )
        assert kept == items
        assert suppressed == []

    def test_missing_decisions_keep_fail_safe(self):
        """(c) Indices past the end of `decisions` are KEPT (fail-safe)."""
        from orchestrator.review_suggestions.prior_round import SETTLED
        items = [_issue('suggestion', f's{i}') for i in range(3)]
        # Only the first suggestion has a decision; indices 1 and 2 are missing.
        kept, suppressed = self._fn()(items, [SETTLED])
        assert kept == items[1:]
        assert suppressed == [items[0]]

    def test_empty_current_returns_empty_pair(self):
        """(d) Empty current → ([], [])."""
        assert self._fn()([], []) == ([], [])
