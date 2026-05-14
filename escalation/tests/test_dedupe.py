"""Tests for escalation/dedupe.py — dedupe helpers and configuration."""

from __future__ import annotations

import json

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


class TestEscalationDedupeFields:
    """Escalation dataclass gains dedupe_count and dedupe_children fields."""

    def _make_min_escalation(self):
        from escalation.models import Escalation
        return Escalation(
            id='esc-1-1',
            task_id='1',
            agent_role='implementer',
            severity='blocking',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )

    def test_defaults_are_zero_and_empty(self):
        """New Escalation has dedupe_count==0 and dedupe_children==[]."""
        esc = self._make_min_escalation()
        assert esc.dedupe_count == 0
        assert esc.dedupe_children == []

    def test_round_trips_via_json(self):
        """dedupe_count and dedupe_children survive to_json / from_json."""
        from escalation.models import Escalation
        esc = self._make_min_escalation()
        esc.dedupe_count = 3
        esc.dedupe_children = ['esc-2-1', 'esc-3-1', 'esc-4-1']
        restored = Escalation.from_json(esc.to_json())
        assert restored.dedupe_count == 3
        assert restored.dedupe_children == ['esc-2-1', 'esc-3-1', 'esc-4-1']

    def test_from_dict_without_dedupe_keys_uses_defaults(self):
        """Old JSON on disk (without dedupe keys) loads with default values."""
        from escalation.models import Escalation
        old_dict = {
            'id': 'esc-1-1',
            'task_id': '1',
            'agent_role': 'implementer',
            'severity': 'blocking',
            'category': 'infra_issue',
            'summary': 'connection lost',
        }
        esc = Escalation.from_dict(old_dict)
        assert esc.dedupe_count == 0
        assert esc.dedupe_children == []

    def test_separate_instances_do_not_share_dedupe_children(self):
        """Two Escalation instances must NOT share the same dedupe_children list."""
        from escalation.models import Escalation
        esc_a = self._make_min_escalation()
        esc_b = self._make_min_escalation()
        esc_a.dedupe_children.append('esc-2-1')
        assert esc_b.dedupe_children == [], (
            'dedupe_children must use default_factory, not a shared class-level list'
        )
