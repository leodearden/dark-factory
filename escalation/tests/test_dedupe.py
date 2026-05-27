"""Tests for escalation/dedupe.py — dedupe helpers and configuration."""

from __future__ import annotations


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

    def test_unicode_punctuation_stripped(self):
        """En-dash, em-dash, and curly quotes are stripped like ASCII punctuation."""
        from escalation.dedupe import summary_dedupe_key

        # ASCII hyphen (Pd): "fused-memory connection timeout"
        k_ascii = summary_dedupe_key('fused-memory connection timeout')
        # En-dash U+2013 (Pd): "fused–memory connection timeout"
        k_en_dash = summary_dedupe_key('fused–memory connection timeout')
        # Em-dash U+2014 (Pd): "fused—memory connection timeout"
        k_em_dash = summary_dedupe_key('fused—memory connection timeout')
        # Curly double quotes U+201C/U+201D (Pi/Pf): "fused“memory” connection timeout"
        k_curly = summary_dedupe_key('fused“memory” connection timeout')

        # All four variants must produce the same normalised key
        assert k_ascii == k_en_dash == k_em_dash == k_curly
        assert k_ascii == ('fusedmemory', 'connection', 'timeout')

    def test_underscore_preserved_in_word_token(self):
        """Underscore (U+005F, category Pc) is NOT stripped — it is part of \\w.

        Deliberate divergence from the previous _PUNCT_TABLE implementation,
        which stripped all Unicode Pc characters (connector punctuation).
        The regex [^\\w\\s] keeps '_' because \\w includes [a-zA-Z0-9_].
        In practice escalation summaries do not use underscores, so the
        divergence is harmless; this test pins the chosen behaviour so it
        cannot drift silently.
        """
        from escalation.dedupe import summary_dedupe_key

        key = summary_dedupe_key('fused_memory connection timeout')
        # '_' is part of \w, so 'fused_memory' is kept as a single token
        assert key == ('fused_memory', 'connection', 'timeout')

    def test_unicode_symbols_stripped(self):
        """Unicode symbol categories (Sm/Sc/Sk/So) ARE stripped by [^\\w\\s].

        Deliberate divergence from the previous _PUNCT_TABLE implementation,
        which only stripped categories starting with 'P' (Pd/Pc/Pi/Pf/Po/Ps/Pe).
        The regex [^\\w\\s] also removes S* characters because none of Sm/Sc/Sk/So
        belong to \\w or \\s.  For example, the math-plus U+002B (Sm category) in
        'cpu+memory' is stripped, merging the two words into 'cpumemory'.
        This test pins the chosen behaviour across two symbol subcategories
        (Sm, Sc) using two examples so any future narrowing of the regex
        is caught immediately.
        """
        from escalation.dedupe import summary_dedupe_key

        # Sm (math symbol): '+' U+002B — stripped, adjacent words merge
        assert summary_dedupe_key('cpu+memory leak') == ('cpumemory', 'leak')

        # Sc (currency symbol): '$' U+0024 — stripped, no token merge here
        assert summary_dedupe_key('cost$ rises today') == ('cost', 'rises', 'today')


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
        esc_a = self._make_min_escalation()
        esc_b = self._make_min_escalation()
        esc_a.dedupe_children.append('esc-2-1')
        assert esc_b.dedupe_children == [], (
            'dedupe_children must use default_factory, not a shared class-level list'
        )


class TestFindDedupeParent:
    """find_dedupe_parent() — scans live queue, returns oldest matching parent id or None."""

    def _make_infra_esc(
        self,
        esc_id: str,
        task_id: str = '1',
        summary: str = 'fused-memory connection timeout on port 8002',
        category: str = 'infra_issue',
    ):
        from escalation.models import Escalation
        return Escalation(
            id=esc_id,
            task_id=task_id,
            agent_role='implementer',
            severity='blocking',
            category=category,
            summary=summary,
        )

    def test_matching_parent_returns_parent_id(self, tmp_path):
        """(a) Same category + first-3-words match within window -> returns parent id."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1', task_id='42')
        queue.submit(parent)

        candidate = self._make_infra_esc(
            'esc-1-2',
            task_id='42',
            summary='fused-memory connection timeout on port 9999',
        )
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result == 'esc-1-1'

    def test_different_category_returns_none(self, tmp_path):
        """(b) Different category (risk_identified vs infra_issue) -> None."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1', task_id='42', category='risk_identified')
        queue.submit(parent)

        candidate = self._make_infra_esc('esc-1-2', task_id='42', category='infra_issue')
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result is None

    def test_different_summary_tokens_returns_none(self, tmp_path):
        """(c) Different first-3-words summary -> None."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1', summary='neo4j connection timeout on port 8002')
        queue.submit(parent)

        candidate = self._make_infra_esc('esc-1-2', summary='fused-memory connection timeout on port 8002')
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result is None

    def test_outside_window_returns_none(self, tmp_path):
        """(d) candidate.timestamp - parent.timestamp > window_secs -> None."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1')
        queue.submit(parent)

        # Move 'now' far into the future, beyond the 600s window
        now = datetime.now(UTC) + timedelta(seconds=700)
        config = DedupeConfig(infra_dedupe_window_secs=600.0)
        candidate = self._make_infra_esc('esc-1-2', summary='fused-memory connection timeout on port 9999')
        result = find_dedupe_parent(queue, candidate, config, now=now)
        assert result is None

    def test_resolved_parent_not_found(self, tmp_path):
        """(e) Already-resolved parent (in archive) -> None, since get_pending() skips archive."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1')
        queue.submit(parent)
        queue.resolve('esc-1-1', 'fixed')

        candidate = self._make_infra_esc('esc-1-2')
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result is None

    def test_cross_task_dedupe(self, tmp_path):
        """(f) Cross-task: parent has task_id='42', candidate has task_id='99' -> returns parent id."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-42-1', task_id='42')
        queue.submit(parent)

        candidate = self._make_infra_esc(
            'esc-99-1',
            task_id='99',
            summary='fused-memory connection timeout on port 9999',
        )
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result == 'esc-42-1'

    def test_multiple_matching_returns_oldest(self, tmp_path):
        """(g) Multiple matching pending parents: returns the OLDEST by timestamp."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        # Older parent (earlier timestamp)
        older = self._make_infra_esc('esc-1-1', task_id='1')
        older.timestamp = (datetime.now(UTC) - timedelta(seconds=60)).isoformat()
        queue.submit(older)

        # Newer parent (later timestamp, still within window)
        newer = self._make_infra_esc('esc-2-1', task_id='2')
        newer.timestamp = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
        queue.submit(newer)

        candidate = self._make_infra_esc('esc-3-1', task_id='3',
                                          summary='fused-memory connection timeout on port 9999')
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result == 'esc-1-1'  # oldest

    def test_oldest_match_with_close_timestamps(self, tmp_path):
        """(h2) Three parents with sub-second gaps inserted out of order: oldest wins.

        Pins that selection uses timestamp comparison, not insertion order or
        filesystem iteration order — a gap the existing test_multiple_matching_returns_oldest
        does not unambiguously cover (it inserts in chronological order with 50s gaps).
        """
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        now = datetime.now(UTC)

        # Submit in non-chronological order: middle, oldest, newest
        middle = self._make_infra_esc('esc-2-1', task_id='2')
        middle.timestamp = (now - timedelta(milliseconds=300)).isoformat()
        queue.submit(middle)

        oldest = self._make_infra_esc('esc-1-1', task_id='1')
        oldest.timestamp = (now - timedelta(milliseconds=500)).isoformat()
        queue.submit(oldest)

        newest = self._make_infra_esc('esc-3-1', task_id='3')
        newest.timestamp = (now - timedelta(milliseconds=100)).isoformat()
        queue.submit(newest)

        candidate = self._make_infra_esc(
            'esc-4-1',
            task_id='4',
            summary='fused-memory connection timeout on port 9999',
        )
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now + timedelta(seconds=5))
        assert result == 'esc-1-1'  # oldest by timestamp, not insertion order

    def test_enabled_flag_not_checked_inside_find_dedupe_parent(self, tmp_path):
        """(h) infra_dedupe_enabled=False does NOT affect find_dedupe_parent itself.

        The gate lives in the server callers, not here. This test pins that contract:
        even with enabled=False, find_dedupe_parent still returns a match.
        """
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1')
        queue.submit(parent)

        candidate = self._make_infra_esc('esc-1-2',
                                          summary='fused-memory connection timeout on port 9999')
        config = DedupeConfig(infra_dedupe_enabled=False)  # disabled
        now = datetime.now(UTC) + timedelta(seconds=5)
        # Despite enabled=False, find_dedupe_parent still finds a match
        result = find_dedupe_parent(queue, candidate, config, now=now)
        assert result == 'esc-1-1'


class TestEscalationDedupeFingerprint:
    """Escalation.dedupe_fingerprint field — added for content-fingerprint dedup (A7a)."""

    def _make_min_escalation(self):
        from escalation.models import Escalation
        return Escalation(
            id='esc-1-1',
            task_id='1',
            agent_role='implementer',
            severity='blocking',
            category='recon_integrity_issue',
            summary='Unresolved after remediation: entity mismatch',
        )

    def test_default_is_none(self):
        """(a) A freshly constructed Escalation has dedupe_fingerprint is None by default."""
        esc = self._make_min_escalation()
        assert esc.dedupe_fingerprint is None

    def test_round_trips_via_json(self):
        """(b) When set to a string, dedupe_fingerprint survives to_json/from_json round-trip."""
        from escalation.models import Escalation
        esc = self._make_min_escalation()
        esc.dedupe_fingerprint = 'abc123deadbeef'
        restored = Escalation.from_json(esc.to_json())
        assert restored.dedupe_fingerprint == 'abc123deadbeef'

    def test_from_dict_without_key_defaults_to_none(self):
        """(c) from_dict on a legacy dict WITHOUT the key defaults to None."""
        from escalation.models import Escalation
        old_dict = {
            'id': 'esc-1-1',
            'task_id': '1',
            'agent_role': 'implementer',
            'severity': 'blocking',
            'category': 'recon_integrity_issue',
            'summary': 'Unresolved after remediation: entity mismatch',
        }
        esc = Escalation.from_dict(old_dict)
        assert esc.dedupe_fingerprint is None

    def test_separate_instances_do_not_share_state(self):
        """(d) Two separate instances do not share the dedupe_fingerprint field."""
        esc_a = self._make_min_escalation()
        esc_b = self._make_min_escalation()
        esc_a.dedupe_fingerprint = 'fingerprint-for-a'
        assert esc_b.dedupe_fingerprint is None, (
            'Setting dedupe_fingerprint on one instance must not affect another'
        )
