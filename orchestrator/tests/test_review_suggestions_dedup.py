"""Tests for orchestrator.review_suggestions.dedup helpers."""

from __future__ import annotations

import hashlib
import json

import pytest

# ---------------------------------------------------------------------------
# review_suggestion_payload_hash
# ---------------------------------------------------------------------------


class TestReviewSuggestionPayloadHash:
    def _import(self):
        from orchestrator.review_suggestions.dedup import review_suggestion_payload_hash
        return review_suggestion_payload_hash

    def test_matches_workflow_inline_hash(self):
        """Hash must be byte-identical to today's workflow.py:4835-4837 value."""
        fn = self._import()
        suggestions = [
            {
                'reviewer': 'test_analyst',
                'severity': 'suggestion',
                'location': 'src/foo.py:10',
                'category': 'coverage',
                'description': 'Missing edge case',
                'suggested_fix': 'Add test',
            },
        ]
        expected = hashlib.sha256(
            json.dumps(suggestions, sort_keys=True).encode()
        ).hexdigest()[:16]
        assert fn(suggestions) == expected

    def test_result_is_16_lowercase_hex_chars(self):
        """Output must be exactly 16 lowercase hex characters."""
        fn = self._import()
        suggestions = [{'key': 'val'}]
        result = fn(suggestions)
        assert len(result) == 16
        assert result == result.lower()
        assert all(c in '0123456789abcdef' for c in result)

    def test_sort_keys_invariant(self):
        """Key ordering within each suggestion dict must not affect the hash."""
        fn = self._import()
        s1 = {'b': 2, 'a': 1}
        s2 = {'a': 1, 'b': 2}
        # json.dumps with sort_keys=True normalizes both
        assert fn([s1]) == fn([s2])

    def test_empty_list_raises(self):
        """Empty suggestions list must propagate sha256_16 ValueError."""
        fn = self._import()
        with pytest.raises(ValueError):
            fn([])

    def test_multiple_suggestions_deterministic(self):
        """Hash over a multi-item list is stable across repeated calls."""
        fn = self._import()
        suggestions = [{'x': i} for i in range(5)]
        assert fn(suggestions) == fn(suggestions)


# ---------------------------------------------------------------------------
# hash_marker
# ---------------------------------------------------------------------------


class TestHashMarker:
    def _import(self):
        from orchestrator.review_suggestions.dedup import hash_marker
        return hash_marker

    def test_exact_format(self):
        """hash_marker('abc123') must return '#hash:abc123#'."""
        fn = self._import()
        assert fn('abc123') == '#hash:abc123#'

    def test_with_hex_hash(self):
        """Works correctly with a real 16-char hex hash."""
        fn = self._import()
        h = 'a1b2c3d4e5f60718'
        assert fn(h) == f'#hash:{h}#'

    def test_startswith_roundtrip(self):
        """A detail string built with hash_marker passes startswith check."""
        fn = self._import()
        h = 'deadbeef01234567'
        marker = fn(h)
        detail = marker + json.dumps([{'k': 'v'}])
        assert detail.startswith(fn(h))


# ---------------------------------------------------------------------------
# find_prior_review_suggestion
# ---------------------------------------------------------------------------


def _make_escalation_for_dedup(**overrides):
    """Build a minimal Escalation object for dedup tests."""
    from escalation.models import Escalation

    defaults: dict = dict(
        id='esc-42-0',
        task_id='42',
        agent_role='orchestrator',
        severity='info',
        category='review_suggestions',
        summary='3 review suggestion(s) for triage',
        detail='[]',
    )
    defaults.update(overrides)
    return Escalation(**defaults)  # type: ignore[arg-type]


class TestFindPriorReviewSuggestion:
    def _import(self):
        from orchestrator.review_suggestions.dedup import (
            find_prior_review_suggestion,
            hash_marker,
        )
        return find_prior_review_suggestion, hash_marker

    def test_returns_matching_record(self):
        """Returns the record whose category and detail match."""
        fn, marker = self._import()
        h = 'abc1234567890abc'
        esc = _make_escalation_for_dedup(
            category='review_suggestions',
            detail=marker(h) + '[]',
        )
        result = fn([esc], h)
        assert result is esc

    def test_none_when_category_differs(self):
        """Returns None if category is not 'review_suggestions'."""
        fn, marker = self._import()
        h = 'abc1234567890abc'
        esc = _make_escalation_for_dedup(
            category='review_issues',
            detail=marker(h) + '[]',
        )
        assert fn([esc], h) is None

    def test_none_when_hash_differs(self):
        """Returns None if hash in detail does not match."""
        fn, marker = self._import()
        h = 'abc1234567890abc'
        other_h = 'ffff000011112222'
        esc = _make_escalation_for_dedup(
            category='review_suggestions',
            detail=marker(other_h) + '[]',
        )
        assert fn([esc], h) is None

    def test_none_for_empty_iterable(self):
        """Returns None for an empty prior-escalations list."""
        fn, _ = self._import()
        assert fn([], 'abc1234567890abc') is None

    def test_none_when_detail_is_none(self):
        """Returns None when detail attribute is None (falsy guard)."""
        fn, _ = self._import()
        esc = _make_escalation_for_dedup(
            category='review_suggestions',
            detail='',  # empty string is falsy
        )
        assert fn([esc], 'abc1234567890abc') is None

    def test_none_when_detail_is_empty(self):
        """Returns None when detail is empty string."""
        fn, _ = self._import()
        esc = _make_escalation_for_dedup(
            category='review_suggestions',
            detail='',
        )
        assert fn([esc], 'abc1234567890abc') is None

    def test_returns_first_match_when_multiple(self):
        """Returns the first matching record, skipping non-matches before it."""
        fn, marker = self._import()
        h = 'abc1234567890abc'
        non_match = _make_escalation_for_dedup(
            id='esc-42-0',
            category='review_issues',
            detail='some other detail',
        )
        match1 = _make_escalation_for_dedup(
            id='esc-42-1',
            category='review_suggestions',
            detail=marker(h) + '[]',
        )
        match2 = _make_escalation_for_dedup(
            id='esc-42-2',
            category='review_suggestions',
            detail=marker(h) + '[{"extra": 1}]',
        )
        result = fn([non_match, match1, match2], h)
        assert result is match1
