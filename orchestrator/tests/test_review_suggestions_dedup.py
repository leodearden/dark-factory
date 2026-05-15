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
