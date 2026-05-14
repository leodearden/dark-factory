"""Tests for escalation/dedupe.py — dedupe helpers and configuration."""

from __future__ import annotations

import pytest


class TestSummaryDedupeKey:
    """summary_dedupe_key() — pure helper, no I/O."""

    def test_similar_summaries_share_key(self):
        """Two infra-style summaries with the same first 3 words share a key."""
        from escalation.dedupe import summary_dedupe_key

        k1 = summary_dedupe_key('Fused-memory  CONNECTION timeout!')
        k2 = summary_dedupe_key('fused-memory connection timeout on port 8002')
        assert k1 == k2

    def test_short_summary_key_length(self):
        """Fewer than 3 tokens → key length equals token count."""
        from escalation.dedupe import summary_dedupe_key

        key = summary_dedupe_key('lost link')
        assert key == ('lost', 'link')
        assert len(key) == 2

    def test_empty_summary_produces_empty_key(self):
        """Empty or whitespace-only summary → empty tuple."""
        from escalation.dedupe import summary_dedupe_key

        assert summary_dedupe_key('') == ()
        assert summary_dedupe_key('   ') == ()
        assert summary_dedupe_key('\t\n') == ()

    def test_different_first_three_tokens_differ(self):
        """Summaries with different first 3 tokens must NOT share a key."""
        from escalation.dedupe import summary_dedupe_key

        k1 = summary_dedupe_key('fused-memory connection timeout on port 8002')
        k2 = summary_dedupe_key('neo4j connection timeout on port 8002')
        assert k1 != k2

    def test_single_token_key(self):
        """Single-word summary → 1-tuple."""
        from escalation.dedupe import summary_dedupe_key

        assert summary_dedupe_key('oops') == ('oops',)

    def test_punctuation_stripped(self):
        """Punctuation is stripped before tokenizing."""
        from escalation.dedupe import summary_dedupe_key

        k1 = summary_dedupe_key('db! connection? lost.')
        k2 = summary_dedupe_key('db connection lost')
        assert k1 == k2

    def test_casefold_applied(self):
        """Key is case-insensitive."""
        from escalation.dedupe import summary_dedupe_key

        k1 = summary_dedupe_key('UPPER lower MiXeD')
        k2 = summary_dedupe_key('upper lower mixed')
        assert k1 == k2
